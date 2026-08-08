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


async def one_request(session, url, model, row, idx, is_warmup, results):
    body = {
        "model": model,
        "messages": row["messages"],
        "max_tokens": row.get("max_tokens", 128),
        "temperature": row.get("temperature", 0.0),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if row.get("ignore_eos"):
        body["ignore_eos"] = True

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
    results.append({
        "request_id": row.get("request_id", idx),
        "index": idx,
        "ttft_s": ttft,
        "itls_s": itls,
        "e2e_s": end - start,
        "output_chunks": len(token_times),
        "usage": usage,
        "constructed_cacheable_fraction": row.get("constructed_cacheable_fraction"),
        "error": error,
    })


async def run(args) -> dict:
    rows = load_requests(Path(args.jsonl), args.warmup + args.num_requests)
    if len(rows) < args.warmup + 1:
        raise SystemExit(f"{args.jsonl}: only {len(rows)} requests available")

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    results = []
    rng = random.Random(args.seed)

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=0)
    offered = 0
    wall_start = time.perf_counter()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if args.concurrency == 1:
            for idx, row in enumerate(rows):
                if _done(idx, args, wall_start):
                    break
                await one_request(session, url, args.model, row, idx,
                                  idx < args.warmup, results)
                offered += 1
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
                                idx < args.warmup, results)))
                offered += 1
            if tasks:
                await asyncio.gather(*tasks)

    wall = time.perf_counter() - wall_start
    return summarize(args, results, offered, wall)


def _done(idx: int, args, wall_start: float) -> bool:
    """Run length: num_requests or min_seconds, whichever is longer."""
    measured = idx - args.warmup
    if measured < args.num_requests:
        return False
    return (time.perf_counter() - wall_start) >= args.min_seconds


def summarize(args, results, offered, wall) -> dict:
    ok = [r for r in results if r["error"] is None and r["ttft_s"] is not None]
    ttfts = [r["ttft_s"] for r in ok]
    itls = [v for r in ok for v in r["itls_s"]]
    out_chunks = sum(r["output_chunks"] for r in ok)
    measured_offered = max(0, offered - args.warmup)
    completion = len(ok) / measured_offered if measured_offered else 0.0

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
        "output_chunk_throughput": round(out_chunks / wall, 3) if wall else None,
        "ttft_s": {
            "p50": percentile(ttfts, 50),
            "p90": percentile(ttfts, 90),
            "p99": percentile(ttfts, 99),
            "mean": statistics.fmean(ttfts) if ttfts else None,
        },
        "itl_s": {
            "p50": percentile(itls, 50),
            "p99": percentile(itls, 99),
            "mean": statistics.fmean(itls) if itls else None,
        },
        # SPEC section 6: below 0.95 the run measures queueing, not caching.
        "tags": ["saturated"] if completion < 0.95 else ["ok"],
    }
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
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--request-timeout", type=float, default=600.0)
    args = p.parse_args()

    payload = asyncio.run(run(args))
    out_path = Path(args.out) / f"{args.run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    s = payload["summary"]
    print(json.dumps({k: s[k] for k in (
        "run_id", "completed", "offered", "completion_ratio", "ttft_s", "tags")},
        indent=2))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
