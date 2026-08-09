#!/usr/bin/env python3
"""Gate A, at a generation length that can actually falsify it.

The original Gate A generated ~36 tokens and reported vLLM PASS / SGLang FAIL.
That asymmetry is not trustworthy: neither engine uses batch-invariant kernels
by default, and reduction-order drift only becomes visible when a softmax
near-tie flips an argmax. Thinking Machines Lab measured 1000 temperature-0 vLLM
completions producing 80 unique outputs that were nonetheless identical for
their first 102 tokens. A 36-token probe sits inside that window, so "identical"
there means "we did not generate far enough to see the flip".

This runs the same equivalence check with a forced long generation, and
optionally under concurrency, which is the other axis batch-invariance covers
(batch composition changes reduction order too).

    gate_a_extended.py --base-url http://localhost:8000 --engine vllm \
        --max-tokens 512 --concurrency 8

Reports, per engine, the token index where the reuse path first diverges from
the fresh-prefill path - or PASS if it never does.
"""

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def send(base_url: str, model: str, messages, max_tokens: int, timeout: float):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # ignore_eos forces the full budget: natural stopping would cut the
        # generation short and hide a late divergence, which is the whole point.
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return (resp["choices"][0]["message"]["content"],
            time.perf_counter() - start,
            resp.get("usage", {}))


def divergence(a: str, b: str, tok=None):
    """First differing index, by token when a tokenizer is available."""
    if tok is not None:
        ta, tb = (tok.encode(a, add_special_tokens=False),
                  tok.encode(b, add_special_tokens=False))
        idx = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), None)
        if idx is None and len(ta) != len(tb):
            idx = min(len(ta), len(tb))
        return idx, len(ta), len(tb), "tokens"
    idx = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    if idx is None and len(a) != len(b):
        idx = min(len(a), len(b))
    return idx, len(a), len(b), "chars"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--requests", default="workloads/full-reuse/requests.jsonl")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--concurrency", type=int, default=0,
                   help="also run this many identical requests at once")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--tokenizer", help="for token-level divergence indices")
    p.add_argument("--out", default="")
    args = p.parse_args()

    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as exc:  # noqa: BLE001
            print(f"(tokenizer unavailable, comparing by character: {exc})")

    messages = json.loads(open(args.requests).readline())["messages"]

    # Serial: first send populates the prefix cache, the next two reuse it.
    cold, t_cold, u_cold = send(args.base_url, args.model, messages,
                                args.max_tokens, args.timeout)
    warm, t_warm, _ = send(args.base_url, args.model, messages,
                           args.max_tokens, args.timeout)
    warm2, _, _ = send(args.base_url, args.model, messages,
                       args.max_tokens, args.timeout)

    # Generated length must come from the server's usage block. Re-tokenizing the
    # decoded text client-side does NOT round-trip to the same count, and reading
    # it that way once produced a spurious "vLLM ignored ignore_eos" conclusion
    # (461/464 of a 512 budget) when the server had in fact emitted exactly 512.
    served = u_cold.get("completion_tokens")
    result = {"engine": args.engine, "max_tokens": args.max_tokens,
              "ttft_like": {"cold_s": t_cold, "warm_s": t_warm,
                            "ratio": (t_warm / t_cold) if t_cold else None},
              "usage_cold": u_cold,
              "server_completion_tokens": served,
              "budget_honoured": (served == args.max_tokens
                                  if served is not None else None)}

    print(f"=== Gate A extended [{args.engine}] max_tokens={args.max_tokens}")
    print(f"  cold {t_cold:.2f}s -> warm {t_warm:.2f}s "
          f"(ratio {t_warm / t_cold:.3f}; reuse active if < 1)")
    print(f"  server completion_tokens={served} "
          f"({'budget honoured' if served == args.max_tokens else 'SHORT'})")

    idx, la, lb, unit = divergence(cold, warm, tok)
    result["cold_vs_warm"] = {"diverges_at": idx, "len_cold": la, "len_warm": lb,
                              "unit": unit}
    if idx is None:
        print(f"  cold == warm  : IDENTICAL over {la} {unit}")
    else:
        print(f"  cold vs warm  : DIVERGES at {unit[:-1]} {idx} of {la}/{lb}")

    idx2, _, _, _ = divergence(warm, warm2, tok)
    result["warm_vs_warm2"] = {"diverges_at": idx2}
    print(f"  warm == warm2 : {'IDENTICAL' if idx2 is None else f'diverges at {idx2}'}"
          "   (reuse path self-consistent?)")

    if args.concurrency > 1:
        # Batch composition is the other axis batch-invariance covers: identical
        # requests sharing a batch can reduce in a different order than alone.
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            outs = [f.result()[0] for f in
                    [pool.submit(send, args.base_url, args.model, messages,
                                 args.max_tokens, args.timeout)
                     for _ in range(args.concurrency)]]
        uniq = len(set(outs))
        vs_serial = divergence(warm, outs[0], tok)[0]
        result["concurrent"] = {"n": args.concurrency, "unique_outputs": uniq,
                                "batched_vs_serial_diverges_at": vs_serial}
        print(f"  concurrency {args.concurrency}: {uniq} unique output(s) among "
              f"{args.concurrency} identical requests")
        print(f"  batched vs serial: "
              f"{'identical' if vs_serial is None else f'diverges at {vs_serial}'}")

    passed = result["cold_vs_warm"]["diverges_at"] is None
    result["verdict"] = "PASS" if passed else "FAIL"
    print(f"  VERDICT: {result['verdict']}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"  -> {args.out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
