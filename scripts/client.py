#!/usr/bin/env python3
"""Fallback async load generator (P0).

Written in P0 so that a Gate C failure cannot stall the day, and reused on day 2
for Mooncake replay if the native path is blocked (P6). Fixed request list in,
per-request timestamps out.

    python3 scripts/client.py --jsonl workloads/W1.jsonl \
        --base-url http://localhost:8000 --model <id> \
        --rate 4.0 --run-id W1_vllm_r4.0 --out results/

Arrivals are open-loop Poisson by default: requests are issued on schedule
regardless of whether earlier ones have completed. SPEC section 6 relies on this
- a closed loop would throttle itself and hide exactly the queueing behaviour
H4b is about. --concurrency 1 switches to the closed loop W-A wants.
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import aiohttp


def percentile(values, q: float):
    """Nearest-rank percentile. At n=200, p99 is the second-worst sample -
    descriptive only, per SPEC section 6."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[idx]


def load_requests(path: Path, limit: int):
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


async def one_request(session, url, model, row, idx, is_warmup, results, image_tokens=None):
    budget = row.get("max_tokens", 128)
    body = {
        "model": model,
        "messages": row["messages"],
        "max_tokens": budget,
        "temperature": row.get("temperature", 0.0),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if row.get("ignore_eos"):
        # Belt and suspenders. ignore_eos alone ignores only the tokenizer's
        # eos_token_id, not the chat template's extra stop tokens - vLLM was
        # measured returning 461/464 of a 512-token budget because the model
        # emitted Gemma's <end_of_turn>. A short run is a fast run, so an
        # unpinned budget silently turns a stopping difference into a latency
        # difference and attributes it to caching.
        body["ignore_eos"] = True
        body["min_tokens"] = budget
        body["stop_token_ids"] = []

    start = time.perf_counter()
    ttft = None
    token_times = []
    usage = None
    error = None

    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                error = f"http {resp.status}: {(await resp.text())[:200]}"
            else:
                async for raw in resp.content:
                    chunk = raw.decode("utf-8", "replace").strip()
                    if not chunk.startswith("data:"):
                        continue
                    payload = chunk[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - start
                        token_times.append(now)
    except Exception as exc:  # noqa: BLE001 - a failed request is data, not a crash
        error = f"{type(exc).__name__}: {exc}"

    end = time.perf_counter()
    if is_warmup:
        return

    itls = [b - a for a, b in zip(token_times, token_times[1:])]
    # SGLang reports per-request cached tokens here when started with
    # --enable-cache-report (non-streaming only for the details block on some
    # versions). vLLM leaves it None and must be attributed from /metrics
    # counter diffs instead. Recording whatever is present keeps Gate R's
    # "genuine cache hit" check per-request rather than per-run.
    details = (usage or {}).get("prompt_tokens_details") or {}
    completed_tokens = (usage or {}).get("completion_tokens")
    prompt_tokens = (usage or {}).get("prompt_tokens")
    # Image-vs-text token split. SGLang reports usage.prompt_tokens_details.
    # image_tokens directly when --enable-cache-report is set; vLLM never
    # populates prompt_tokens_details (confirmed None across every P1 probe).
    # But image token count is a workload CONSTANT - fixed resolution, so fixed
    # ViT output size (SPEC section 5; measured 258 for every resolution tried
    # in P0's probe) - so it can be supplied once from the manifest and applied
    # uniformly to both engines rather than depended on per-response.
    reported_image_tokens = details.get("image_tokens")
    image_tok = reported_image_tokens if reported_image_tokens is not None else image_tokens
    text_tok = (prompt_tokens - image_tok
               if prompt_tokens is not None and image_tok is not None else None)
    dur = end - start
    results.append({
        "request_id": row.get("request_id", idx),
        "index": idx,
        "ttft_s": ttft,
        "itls_s": itls,
        "e2e_s": dur,
        "output_chunks": len(token_times),
        "usage": usage,
        "completion_tokens": completed_tokens,
        "budget_honoured": (completed_tokens == row.get("max_tokens")
                            if completed_tokens is not None else None),
        "cached_tokens": details.get("cached_tokens"),
        "prompt_tokens": prompt_tokens,
        "image_tokens": image_tok,
        "text_tokens": text_tok,
        "tokens_per_second_this_request": (completed_tokens / dur
                                          if completed_tokens and dur else None),
        "constructed_cacheable_fraction": row.get("constructed_cacheable_fraction"),
        "error": error,
    })


async def run(args) -> dict:
    # _done() enforces "num_requests OR min_seconds, whichever is LONGER" - but
    # that only works if enough rows are pre-sliced to sustain arrivals for the
    # full min_seconds at this --rate. load_requests() used to slice exactly
    # warmup+num_requests rows regardless of rate, so at any rate fast enough to
    # deliver num_requests within min_seconds, the loop ran out of rows and
    # ended early - silently, with no tag, reporting a truncated run as clean
    # `ok`. At the real P2 grid's higher rates (e.g. 4.9 req/s) the default
    # 250 rows arrive in ~51s against an intended 240s floor. pilot.py already
    # solved this correctly (rate * duration * 1.1 headroom); this carries the
    # same fix into the path every other caller (run.sh cell(), by extension
    # the whole P3 matrix) actually exercises, rather than requiring every
    # caller to separately remember to pass a rate-scaled --num-requests.
    min_needed = args.warmup + args.num_requests
    if args.concurrency != 1 and args.rate > 0:
        min_needed = max(min_needed,
                         args.warmup + int(args.rate * args.min_seconds * 1.15))
    rows = load_requests(Path(args.jsonl), min_needed)
    if len(rows) < args.warmup + 1:
        raise SystemExit(f"{args.jsonl}: only {len(rows)} requests available")
    if len(rows) < min_needed:
        print(f"WARNING: wanted {min_needed} rows for rate {args.rate} over "
              f"{args.min_seconds}s, only {len(rows)} available in {args.jsonl} - "
              f"this run may end before min_seconds and understate its true "
              f"duration. Rebuild the workload with a higher --max-rate.",
              file=sys.stderr)

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    results = []
    rng = random.Random(args.seed)

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=0)
    offered = 0
    wall_start = time.perf_counter()

    # Ran dry vs stopped on purpose: `rows` is a hard ceiling `_done()` cannot
    # override. If the loop exhausts every row without `_done()` ever firing,
    # that is exactly the silent-truncation failure the headroom above exists
    # to prevent - and headroom is a margin, not a guarantee, so this must be
    # asserted rather than trusted. for/else runs the else clause only when
    # the loop completes WITHOUT break, which is precisely "ran out of rows."
    rows_exhausted = False

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if args.concurrency == 1:
            for idx, row in enumerate(rows):
                if _done(idx, args, wall_start):
                    break
                await one_request(session, url, args.model, row, idx,
                                  idx < args.warmup, results, args.image_tokens)
                offered += 1
            else:
                rows_exhausted = bool(rows) and not _done(len(rows), args, wall_start)
        else:
            tasks = []
            elapsed = 0.0
            for idx, row in enumerate(rows):
                if _done(idx, args, wall_start):
                    break
                if idx:  # Poisson: exponential inter-arrival gaps
                    elapsed += rng.expovariate(args.rate)
                delay = elapsed - (time.perf_counter() - wall_start)
                if delay > 0:
                    await asyncio.sleep(delay)
                tasks.append(asyncio.create_task(
                    one_request(session, url, args.model, row, idx,
                                idx < args.warmup, results, args.image_tokens)))
                offered += 1
            else:
                rows_exhausted = bool(rows) and not _done(len(rows), args, wall_start)
            if tasks:
                await asyncio.gather(*tasks)

    wall = time.perf_counter() - wall_start
    return summarize(args, results, offered, wall, rows_exhausted)


def _done(idx: int, args, wall_start: float) -> bool:
    """Run length: num_requests or min_seconds, whichever is longer."""
    measured = idx - args.warmup
    if measured < args.num_requests:
        return False
    return (time.perf_counter() - wall_start) >= args.min_seconds


def stats(values):
    if not values:
        return {"p50": None, "p90": None, "p99": None, "mean": None}
    return {"p50": percentile(values, 50), "p90": percentile(values, 90),
            "p99": percentile(values, 99), "mean": statistics.fmean(values)}


def ttft_growth_ratio(ok):
    """Median TTFT of the run's second half over its first half, in arrival
    order. SPEC section 6's own definition of saturation is "open-loop TTFT is
    a function of run length - the queue grows without bound" - completion
    ratio can stay a perfect 1.0 while this is happening (the engine still
    finishes everything, just far too slowly), which is exactly what let half
    of the real P3 matrix's cells collapse into 10-180s TTFT while tagged `ok`.
    This operationalizes SPEC's own sentence directly: a flat run has ratio
    near 1.0; a collapsing one grows without bound. Verified against all 28
    real matrix cells before picking the threshold: every genuinely healthy
    cell measured 1.00-1.07, every visibly collapsed one (already obvious from
    raw TTFT) measured >=2.4 - wide, unambiguous separation, no borderline
    cases sitting near the cutoff."""
    reqs = sorted((r for r in ok), key=lambda r: r["index"])
    n = len(reqs)
    if n < 10:
        return None
    first = [r["ttft_s"] for r in reqs[: n // 2]]
    second = [r["ttft_s"] for r in reqs[n // 2 :]]
    m1 = statistics.median(first)
    return (statistics.median(second) / m1) if m1 > 0 else None


def summarize(args, results, offered, wall, rows_exhausted=False) -> dict:
    ok = [r for r in results if r["error"] is None and r["ttft_s"] is not None]
    ttfts = [r["ttft_s"] for r in ok]
    # ITL/TPOT: average gap between consecutive decoded tokens, pooled across
    # all requests. TTFT is excluded by construction (itls_s only ever holds
    # gaps between token 2..N, never the arrival-to-first-token gap).
    itls = [v for r in ok for v in r["itls_s"]]
    out_chunks = sum(r["output_chunks"] for r in ok)
    measured_offered = max(0, offered - args.warmup)
    completion = len(ok) / measured_offered if measured_offered else 0.0
    ttft_growth = ttft_growth_ratio(ok)

    # Gate R condition 2: if the engine did not honour the budget, this run did
    # a different amount of work than its counterpart and its latency is not
    # comparable. Surfaced as a tag, not buried in per-request rows.
    checked = [r for r in ok if r["budget_honoured"] is not None]
    short = [r for r in checked if not r["budget_honoured"]]
    # A successful (in `ok`) request whose usage.completion_tokens the engine
    # simply omitted is silently excluded from the check above rather than
    # counted - not seen on either engine so far, but worth surfacing rather
    # than letting a run report budget_short_count=0 while a chunk of its
    # requests were never actually verified.
    unchecked = len(ok) - len(checked)
    cached = [r["cached_tokens"] for r in ok if r["cached_tokens"] is not None]
    prompts = [r["prompt_tokens"] for r in ok if r["prompt_tokens"]]

    # Token-counted throughput, not chunk-counted: a streamed SSE chunk is
    # usually one token but is not guaranteed to be, so summing the engine's
    # own usage.completion_tokens is the number that actually matches the
    # blog's TPS = total_output_tokens / (Ty - Tx) definition.
    completion_tokens = [r["completion_tokens"] for r in ok
                         if r["completion_tokens"] is not None]
    per_user_tps = [r["tokens_per_second_this_request"] for r in ok
                    if r["tokens_per_second_this_request"] is not None]
    isl = [r["prompt_tokens"] for r in ok if r["prompt_tokens"] is not None]
    osl = [r["completion_tokens"] for r in ok if r["completion_tokens"] is not None]
    image_toks = [r["image_tokens"] for r in ok if r["image_tokens"] is not None]
    text_toks = [r["text_tokens"] for r in ok if r["text_tokens"] is not None]

    summary = {
        "run_id": args.run_id,
        "jsonl": args.jsonl,
        "engine": args.engine,
        "workload": args.workload,
        "model": args.model,
        "rate": args.rate,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "warmup": args.warmup,
        "offered": measured_offered,
        "completed": len(ok),
        "failed": len(results) - len(ok),
        "completion_ratio": round(completion, 4),
        "wall_seconds": round(wall, 3),
        # System-wide throughput: total output tokens actually generated across
        # every concurrent request, divided by wall time.
        "tokens_per_second": (round(sum(completion_tokens) / wall, 3)
                             if completion_tokens and wall else None),
        # Per-user throughput: mean of each request's own completion_tokens/e2e.
        # Approaches 1/ITL as output length grows (NVIDIA's LLM benchmarking
        # blog, developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts).
        "tokens_per_second_per_user": (round(statistics.fmean(per_user_tps), 3)
                                      if per_user_tps else None),
        "requests_per_second": round(len(ok) / wall, 3) if wall else None,
        "output_chunk_throughput": round(out_chunks / wall, 3) if wall else None,
        "ttft_s": stats(ttfts),
        "itl_s": stats(itls),
        # ISL/OSL: input and output sequence length, the two quantities the
        # blog identifies as driving TTFT (via prefill memory/compute) and ITL
        # (via decode memory bandwidth) respectively.
        "isl_tokens": stats(isl),
        "osl_tokens": stats(osl),
        "image_tokens": stats(image_toks),
        "text_tokens": stats(text_toks),
        # SPEC section 6: below 0.95 the run measures queueing, not caching.
        "budget_honoured_count": len(checked) - len(short),
        "budget_short_count": len(short),
        "budget_unchecked_count": unchecked,
        "cached_tokens_reported": len(cached),
        "measured_cached_fraction": (
            round(sum(cached) / sum(prompts), 4)
            if cached and prompts and sum(prompts) else None),
        # budget_unpinned is a Gate R failure - the compared runs did different
        # amounts of work, so their latencies are not comparable however clean
        # they look. rows_exhausted means the pre-built request file ran out
        # before min_seconds elapsed - the run silently measured LESS time than
        # intended (the headroom math is a margin, not a guarantee, so this is
        # what actually catches it when the margin isn't enough).
        "rows_exhausted": rows_exhausted,
        "ttft_growth_ratio": ttft_growth,
    }
    # Saturated by either signal: completion ratio dropping (the original
    # check) OR TTFT growing without bound within the run (SPEC section 6's
    # own definition, which completion ratio alone can miss entirely - the
    # engine can finish every request and still take 10-180s doing it).
    ratio_saturated = ttft_growth is not None and ttft_growth > 2.0
    summary["tags"] = (
        (["saturated"] if (completion < 0.95 or ratio_saturated) else ["ok"])
        + (["budget_unpinned"] if short else [])
        + (["rows_exhausted"] if rows_exhausted else []))
    return {"summary": summary, "requests": results}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--engine", default="", help="vllm | sglang, recorded in the result")
    p.add_argument("--workload", default="", help="W1 | W2 | W3 | W-A")
    p.add_argument("--rate", type=float, default=1.0, help="offered req/s, open loop")
    p.add_argument("--concurrency", type=int, default=0,
                   help="0 = open loop Poisson; 1 = closed loop, for W-A")
    p.add_argument("--num-requests", type=int, default=200)
    p.add_argument("--min-seconds", type=float, default=240.0)
    p.add_argument("--warmup", type=int, default=50,
                   help="was 20; raised after finding vLLM's own log emitting "
                        "'Triton kernel JIT compilation during inference ... "
                        "consider extending warmup' inside a MEASURED window "
                        "across 10 separate run logs on this pod - 20 requests "
                        "does not reliably touch every batch-size bucket/shape "
                        "that triggers a distinct kernel or captured graph. "
                        "See scripts/check_jit_contamination.py for the per-run "
                        "assertion that now catches it if it still happens.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--image-tokens", type=int, default=None,
                   help="per-image token count from the P0 probe / manifest.json "
                        "geometry.image_tokens - used as the text/image split "
                        "fallback when the engine's own response does not report "
                        "it (true for vLLM on every version probed so far)")
    args = p.parse_args()

    payload = asyncio.run(run(args))
    out_path = Path(args.out) / f"{args.run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    s = payload["summary"]
    print(json.dumps({k: s[k] for k in (
        "run_id", "completed", "offered", "completion_ratio", "ttft_s", "itl_s",
        "tokens_per_second", "tokens_per_second_per_user", "requests_per_second",
        "isl_tokens", "osl_tokens", "tags")},
        indent=2))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
