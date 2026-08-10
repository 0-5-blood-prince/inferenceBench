# Was the vLLM vs SGLang comparison fair? — a post-hoc audit

Done after the full P3 matrix and text-only control ran, by cross-checking
our recorded artifacts (`gates/*/config_dump.json`, Prometheus scrapes in
`workloads/*/metrics/`, engine startup logs captured in
`gates/*/cuda_graph_info.txt`) against both engines' actual source at the
pinned tags (vLLM v0.26.0, SGLang v0.5.16).

**Verdict: fair as a defaults-at-equal-memory benchmark, with two disclosed
structural asymmetries — both of which are engine-shipped behavior for this
model, not configuration mistakes.** The headline gap and SGLang's earlier
saturation are properties of the engines, not of the harness. Details below,
strongest evidence first.

## Verified matched (the comparison stands on these)

- **Workload**: byte-identical `requests.jsonl` per workload, same client
  venv, `temperature=0.0`, pinned output budgets with `ignore_eos` +
  `min_tokens` + empty `stop_token_ids`, same warmup discard, same
  measurement code path for both engines.
- **Scheduling policy**: both FCFS. SGLang's own `/get_server_info` reports
  `schedule_policy = fcfs` (the "SGLang defaults to LPM reordering" concern
  is outdated — default has been fcfs since ≤ v0.4.6; confirmed in the
  v0.5.16 source). vLLM v0.26 default is likewise `fcfs`.
- **Chunked prefill**: on for both (vLLM default-on; SGLang
  `chunked_prefill_size = 8192` on an 80GB card, Gemma-family multimodal is
  not on its disable list).
- **KV-cache capacity — the big fear, measured and dismissed.** The flags'
  semantics genuinely differ (vLLM's `--gpu-memory-utilization 0.90` caps
  weights+activations+KV+graphs *in total*, after a profiling pass; SGLang's
  `--mem-fraction-static 0.90` covers only weights+KV, activations live in
  the other 10%), so equal numbers do not guarantee equal pools. But the
  *measured* outcomes were close: vLLM `kv_cache_size_tokens = 18476` (from
  `vllm:cache_config_info` in our own metrics scrapes) vs SGLang
  `max_total_num_tokens = 16391` — ~13% apart, both tiny (~2 full-length
  8k sequences). Neither engine got a materially bigger cache. SGLang's
  ~16k figure is correct arithmetic for a ~62GB bf16 model in a 72GB static
  budget, not a misconfiguration.
- **Shared-image encoder work — symmetric, both engines cache it.** vLLM:
  4GiB processor cache + encoder cache + prefix-KV-hit encoder skip; our
  scrapes show `vllm:mm_cache_hits_total`/`queries` = 471/472 on Full reuse
  (ViT ran ~once for the whole run). SGLang: 100MB VLM embedding cache on
  by default (`SGLANG_VLM_CACHE_SIZE_MB=100`, hash-keyed, hits skip ViT
  entirely) **and** its radix cache folds the image hash into the token
  prefix (`pad_value = hash`), so identical-image prefixes get KV-level
  hits. The `enable_prefix_mm_cache=False` / `enable_mm_global_cache=False`
  entries in its config dump are *different, cross-process* caches — their
  being off does not mean the shared image was re-encoded. For Cold (unique
  images) both engines re-encode every image. Symmetric on both ends.
- **Metrics/reporting flags**: vLLM's Prometheus metrics are on by default;
  SGLang needed `--enable-metrics --enable-cache-report`. Neither flag has
  any documented per-request overhead (cache-report only *exposes* a count
  that is tracked regardless), and both are standard production settings.
  Not a plausible confound at the millisecond scale we measured.
- **Explicit 0.90 did not secretly favor either side.** For vLLM it is
  *below* the v0.26 default of 0.92 (slight self-handicap, disclosed). For
  SGLang, passing it explicitly *skipped* the VLM auto-reduction
  (~0.95×) that its auto-derived default (~0.86–0.87) would have applied —
  i.e. explicit 0.90 gave SGLang *more* KV than its own defaults would have.

## Disclosed asymmetry 1 — CUDA graph coverage of prefill (already known, now sharpened)

SGLang v0.5.16 auto-disables prefill graph capture for any multimodal model
not on a short allowlist (Gemma 4 isn't on it): "Breakable CUDA graph is
incompatible with multimodal model" — decode graphs stay on. vLLM's default
`-O2` gives `FULL_AND_PIECEWISE`: full graphs for decode batches,
**piecewise graphs still covering prefill and mixed batches** of the LM
backbone. Neither engine graphs or compiles the vision encoder
(`compile_mm_encoder=False` / `cudagraph_mm_encoder=False` on vLLM; ViT is
cache-served on both). So the asymmetry is real but narrower than "graphs
vs no graphs": it is *piecewise-graphed LM prefill (vLLM) vs eager LM
prefill (SGLang)*, on this model, by both engines' own defaults. SGLang
*could* have been forced (`--cuda-graph-backend-prefill ...`) but that path
is explicitly unvalidated for this architecture — forcing it would have
tested a configuration no user runs. Recorded-not-equalized remains the
right call; the [text-only control](../../workloads/full-reuse-text/README.md)
bounds this whole multimodal-prefill bundle at roughly a third of the gap.

## Disclosed asymmetry 2 — prefill admission budgets differ 4–8×, by defaults

vLLM v0.26 carves A100 out of its large-batch default path (throughput
regression, PR #17885): on this GPU it runs `max_num_batched_tokens = 2048`,
`max_num_seqs = 256`. SGLang ran `chunked_prefill_size = 8192`,
`max_prefill_tokens = 16384`. Both are shipped defaults, but they are not
the same knob value, and an H100 rerun of the identical command would give
vLLM 8192/1024 — the comparison is A100-specific in a way the spec's "N=1
hardware" limitation should name explicitly. Direction is ambiguous
(smaller chunks favor TTFT interleaving under decode priority; bigger
chunks favor raw prefill throughput), and vLLM won *despite* the smaller
budget, so this does not undermine the headline direction.

## The mechanism the audit actually surfaced (belongs in P4's mechanism bullet)

vLLM's v1 scheduler is **decode-prioritizing with cheap cached-prefill
drain**: running requests are served first, waiting requests admitted FCFS
into leftover token budget, and — decisive under prefix caching — a waiting
request's admission cost is only its *uncached* tokens. On Full reuse
(~87% cacheable), queued requests are nearly free to admit, so vLLM keeps
TTFT flat at `0.8κ` even as load approaches the knee. SGLang is
prefill-prioritizing but pays full prefill for its queue, and its
optimistic admission against a ~16k-token pool triggers
**retract-and-re-prefill cycles** (retraction evicts running decodes back
to the waiting queue; they re-prefill later) — a positive-feedback loop
under sustained open-loop arrivals. This matches what we measured back in
P2 without knowing why: the pilot preemption counters showed
"preemption thrashing" saturation on Full/Partial reuse
([pilot-results.md](pilot-results.md)), and the P3 `0.8κ` cells collapsed
on SGLang first. **This is an engine-design finding, not an unfairness** —
both engines had near-equal KV; they differ in what they do when it runs
out.

## What this changes downstream

- P4's mechanism bullet gains a third, now best-supported leg: scheduler
  admission/retraction design under small KV, alongside the CUDA-graph and
  cache-path stories. The text-only control's "gap persists on text" residue
  is consistent with the scheduler story (it does not need images).
- Gate B's "partial drop — one cache layer only" reading deserves a
  caveat: both engines demonstrably cache both layers (encoder + KV); a
  partial TTFT drop on the second request does not cleanly imply a missing
  cache layer.
- SPEC's known-limitations list should name the A100-specific vLLM batch
  defaults (asymmetry 2) alongside N=1 hardware.
