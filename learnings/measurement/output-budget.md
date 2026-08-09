# Pinning the output budget — and a measurement error worth recording

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
tokenizer. (The same lesson resurfaced independently in
[pilot-methodology.md](pilot-methodology.md) via a different bug.)

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

**Also affected:** the divergence indices recorded in
[../gate-a/mechanism.md](../gate-a/mechanism.md) (token 55 / token 21) were
computed on the same client-retokenized text. They locate divergence
approximately, which is enough for a diagnostic, but they are not exact server
token positions.
