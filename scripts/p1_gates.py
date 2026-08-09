#!/usr/bin/env python3
"""P1 validation gates (see spec/phases/P1-gates.md).

Proves the measurement is valid before any run is spent on it. Gate B blocks
only if both engines fail; Gate C decides whether the native harness or the P0
fallback client drives the matrix.

    p1_gates.py config-dump --base-url ... --engine vllm
    p1_gates.py gate-b      --base-url ... --model M --engine vllm
    p1_gates.py gate-c      --base-url ... --model M --requests workloads/full-reuse/requests.jsonl

Everything lands in gates/<engine>/ and is committed at P2.

Gate A (strict cold==warm output equivalence) is NOT run by run.sh p1() and is
no longer blocking - see spec/phases/P1-gates.md and LEARNINGS.md sections 1-4.
Measured at a real generation length (512 tokens), both engines fail it, and it
tests a property no latency measurement can see. capture/gate-a below are kept
for manual investigation only. The graded replacement - divergence index at a
stated length, informational - is scripts/gate_a_extended.py, which run.sh p1()
does call.

    # server up with prefix caching DISABLED
    p1_gates.py capture --base-url ... --model M --engine vllm --label cache-off
    # restart with caching ENABLED, then capture twice without restarting
    p1_gates.py capture --base-url ... --model M --engine vllm --label cold
    p1_gates.py capture --base-url ... --model M --engine vllm --label warm
    p1_gates.py gate-a --engine vllm
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Semantics differ per engine; Gate B is what pins which is which, so both
# candidates are recorded rather than assumed comparable.
COUNTER_CANDIDATES = [
    "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
    "vllm:gpu_prefix_cache_hit_rate", "vllm:num_preemptions_total",
    "sglang:cached_tokens_total", "sglang:prompt_tokens_total",
    "sglang:cache_hit_rate", "sglang:num_preemptions_total",
]

CONFIG_ENDPOINTS = ["/get_server_info", "/v1/models", "/get_model_info", "/health"]


def http_json(url: str, payload=None, timeout=300.0):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_text(url: str, timeout=60.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def scrape(base_url: str) -> dict:
    """Snapshot whichever cache/preemption counters this engine exposes."""
    for path in ("/metrics", "/v1/metrics"):
        try:
            text = http_text(base_url.rstrip("/") + path)
        except Exception:  # noqa: BLE001
            continue
        found = {}
        for name in COUNTER_CANDIDATES:
            m = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$",
                          text, re.MULTILINE)
            if m:
                found[name] = float(m.group(1))
        return found
    return {}


def probe_request(model: str, requests_path: Path | None, max_tokens: int) -> dict:
    """One fixed greedy request. Uses a real built request when available so the
    gates exercise the same multimodal path the matrix will."""
    if requests_path and requests_path.exists():
        first = json.loads(requests_path.read_text().splitlines()[0])
        messages = first["messages"]
    else:
        messages = [{"role": "user", "content": "Describe prime numbers briefly."}]
    return {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": False}


def send(base_url: str, body: dict) -> tuple[str, float, dict]:
    start = time.perf_counter()
    resp = http_json(base_url.rstrip("/") + "/v1/chat/completions", body)
    elapsed = time.perf_counter() - start
    return resp["choices"][0]["message"]["content"], elapsed, resp.get("usage", {})


def outdir(engine: str) -> Path:
    path = Path("gates") / engine
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_capture(args) -> None:
    body = probe_request(args.model, Path(args.requests) if args.requests else None,
                         args.max_tokens)
    text, elapsed, usage = send(args.base_url, body)
    record = {"label": args.label, "engine": args.engine, "elapsed_s": elapsed,
              "usage": usage, "output": text}
    path = outdir(args.engine) / f"gate_a_{args.label}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{args.label}: {len(text)} chars, {elapsed:.3f}s -> {path}")


def cmd_gate_a(args) -> None:
    """Reused KV must be indistinguishable from a fresh prefill."""
    folder = outdir(args.engine)
    labels = ("cache-off", "cold", "warm")
    captures = {}
    for label in labels:
        path = folder / f"gate_a_{label}.json"
        if not path.exists():
            sys.exit(f"missing capture {path} - run `capture --label {label}` first")
        captures[label] = json.loads(path.read_text())["output"]

    baseline = captures["cache-off"]
    exact = all(captures[l] == baseline for l in labels)
    prefix_ok = all(captures[l][:args.prefix_chars] == baseline[:args.prefix_chars]
                    for l in labels)

    verdict = "PASS" if exact else ("PASS (prefix only)" if prefix_ok else "FAIL")
    print(f"Gate A [{args.engine}]: {verdict}")
    for label in labels:
        same = "identical" if captures[label] == baseline else "DIVERGES"
        print(f"  {label:10s} {same}")
    if not exact and prefix_ok:
        print(f"  first {args.prefix_chars} chars agree; divergence is tail-only. "
              "Record it and proceed (P1-gates.md).")

    (folder / "gate_a_result.json").write_text(json.dumps(
        {"engine": args.engine, "exact": exact, "prefix_ok": prefix_ok,
         "prefix_chars": args.prefix_chars, "verdict": verdict}, indent=2) + "\n")
    if not prefix_ok:
        sys.exit("Gate A is blocking: reused KV != fresh prefill.")


def cmd_gate_b(args) -> None:
    """The double-send TTFT ratio is what reveals what the multimodal path caches."""
    body = probe_request(args.model, Path(args.requests) if args.requests else None,
                         args.max_tokens)
    pre = scrape(args.base_url)
    _, first, usage1 = send(args.base_url, body)
    mid = scrape(args.base_url)
    _, second, usage2 = send(args.base_url, body)
    post = scrape(args.base_url)

    ratio = second / first if first else float("nan")
    if ratio > 0.8:
        reading = ("no meaningful reuse - the multimodal prefix is not being "
                   "cached by this engine")
    elif ratio <= 0.6:
        reading = "clear reuse engaged"
    else:
        reading = ("partial drop - likely one cache layer only (KV without the "
                   "vision tower, or the reverse). This is a finding, not a failure.")

    print(f"Gate B [{args.engine}]: send1 {first:.3f}s, send2 {second:.3f}s, "
          f"ratio {ratio:.3f}")
    print(f"  {reading}")
    deltas = {k: round(post.get(k, 0) - pre.get(k, 0), 4)
              for k in set(pre) | set(post)}
    print(f"  counter deltas over both sends: {deltas or 'none exposed'}")
    print("  usage.prompt_tokens_details (if present) pins hit-rate semantics: "
          f"{usage1.get('prompt_tokens_details')} -> "
          f"{usage2.get('prompt_tokens_details')}")

    (outdir(args.engine) / "gate_b_result.json").write_text(json.dumps(
        {"engine": args.engine, "ttft1_s": first, "ttft2_s": second,
         "ratio": ratio, "reading": reading,
         "counters": {"pre": pre, "mid": mid, "post": post, "delta": deltas},
         "usage": [usage1, usage2]}, indent=2) + "\n")


def cmd_gate_c(args) -> None:
    """Can the native harness express the workloads, or does the fallback drive?"""
    path = Path(args.requests)
    checks = {}
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()][:2]
    checks["custom jsonl parses"] = len(rows) >= 2

    try:
        body = dict(rows[0], model=args.model, stream=False)
        body.pop("request_id", None)
        body.pop("constructed_cacheable_tokens", None)
        body.pop("constructed_cacheable_fraction", None)
        text, _, _ = send(args.base_url, body)
        checks["image sent through the chat endpoint"] = bool(text)
        text2, _, _ = send(args.base_url, body)
        checks["identical image replayed"] = bool(text2)
    except Exception as exc:  # noqa: BLE001
        checks[f"chat endpoint failed: {type(exc).__name__}"] = False

    for label, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    passed = all(checks.values())
    verdict = ("PASS - native harness can drive the matrix" if passed else
               "FAIL - use the P0 fallback client (scripts/client.py)")
    print(f"Gate C: {verdict}")
    (outdir(args.engine) / "gate_c_result.json").write_text(json.dumps(
        {"engine": args.engine, "checks": checks, "passed": passed}, indent=2) + "\n")


def cmd_config_dump(args) -> None:
    """Commit what each engine actually decided, especially KV capacity."""
    dump = {"engine": args.engine, "base_url": args.base_url, "endpoints": {}}
    for path in CONFIG_ENDPOINTS:
        try:
            dump["endpoints"][path] = http_json(args.base_url.rstrip("/") + path)
        except Exception as exc:  # noqa: BLE001
            dump["endpoints"][path] = f"unavailable: {type(exc).__name__}"
    dump["counters_at_startup"] = scrape(args.base_url)
    out = outdir(args.engine) / "config_dump.json"
    out.write_text(json.dumps(dump, indent=2) + "\n")
    print(f"config dump -> {out}")
    print("Check by hand and record: block size, chunked prefill, scheduler policy, "
          "max_num_seqs, memory fraction, reported KV-token capacity.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, need_model=True):
        sp.add_argument("--base-url", required=True)
        sp.add_argument("--engine", required=True, choices=("vllm", "sglang"))
        if need_model:
            sp.add_argument("--model", required=True)
        sp.add_argument("--requests",
                        default="workloads/full-reuse/requests.jsonl",
                        help="built requests, so gates use the real multimodal path")
        sp.add_argument("--max-tokens", type=int, default=128)

    cap = sub.add_parser("capture", help="one greedy send, saved for Gate A")
    common(cap)
    cap.add_argument("--label", required=True, choices=("cache-off", "cold", "warm"))
    cap.set_defaults(func=cmd_capture)

    ga = sub.add_parser("gate-a", help="compare the three captures")
    ga.add_argument("--engine", required=True, choices=("vllm", "sglang"))
    ga.add_argument("--prefix-chars", type=int, default=256)
    ga.set_defaults(func=cmd_gate_a)

    gb = sub.add_parser("gate-b", help="double-send TTFT ratio + counter semantics")
    common(gb)
    gb.set_defaults(func=cmd_gate_b)

    gc = sub.add_parser("gate-c", help="harness capability")
    common(gc)
    gc.set_defaults(func=cmd_gate_c)

    cd = sub.add_parser("config-dump", help="commit each engine's actual config")
    common(cd, need_model=False)
    cd.set_defaults(func=cmd_config_dump)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
