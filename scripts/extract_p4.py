#!/usr/bin/env python3
"""P4: pull every run's summary into one table, and recover the cached-token
fraction for *both* engines from the Prometheus scrapes.

Why this exists: client.py only records `cached_tokens` when the engine returns
it per-request in the OpenAI response (`usage.prompt_tokens_details`). SGLang
does; vLLM 0.26.0 does not -- so every vLLM run carries
`measured_cached_fraction: None`. That is an instrument gap, not a cache gap:
vLLM publishes the same quantity as Prometheus counters. Since run.sh restarts
the server per run, the pre-scrape is all zeros and the post-scrape *is* the
run's delta, so the fractions below are per-run and directly comparable.

vLLM   : prefix_cache_hits_total / prefix_cache_queries_total  (token-weighted)
SGLang : cached_tokens_total     / prompt_tokens_total          (token-weighted)

Both denominators include the 50 warmup requests, so both numerators do too --
consistent, and noted in the README rather than silently corrected.
"""
import json
import glob
import os
import re
import sys

METRIC_RE = re.compile(r'^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$')

VLLM_KEYS = {
    "hits": "vllm:prefix_cache_hits_total",
    "queries": "vllm:prefix_cache_queries_total",
    "mm_hits": "vllm:mm_cache_hits_total",
    "mm_queries": "vllm:mm_cache_queries_total",
    "prompt": "vllm:prompt_tokens_total",
    "preempt": "vllm:num_preemptions_total",
}
SGLANG_KEYS = {
    "hits": "sglang:cached_tokens_total",
    "queries": "sglang:prompt_tokens_total",
    "prompt": "sglang:prompt_tokens_total",
    "evicted": "sglang:evicted_tokens_total",
    "requests": "sglang:num_requests_total",
}


def scrape(path):
    """Sum every labelled series per metric name (label sets are singletons here)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = METRIC_RE.match(line)
            if not m:
                continue
            name, _labels, val = m.groups()
            if val == "NaN":
                continue
            out[name] = out.get(name, 0.0) + float(val)
    return out


def metrics_path(workload, run_id, which):
    return os.path.join("workloads", workload, "metrics", f"{run_id}_{which}.txt")


def cache_stats(workload, run_id, engine):
    pre = scrape(metrics_path(workload, run_id, "pre"))
    post = scrape(metrics_path(workload, run_id, "post"))
    keys = VLLM_KEYS if engine == "vllm" else SGLANG_KEYS
    d = {}
    for alias, name in keys.items():
        if name in post:
            d[alias] = post[name] - pre.get(name, 0.0)
    frac = None
    if d.get("queries") and "hits" in d:
        frac = d["hits"] / d["queries"]
    mm = None
    if d.get("mm_queries") and "mm_hits" in d:
        mm = d["mm_hits"] / d["mm_queries"]
    return frac, mm, d


def main():
    rows = []
    for f in sorted(glob.glob("workloads/*/results/*.json")):
        s = json.load(open(f))["summary"]
        run_id = s["run_id"]
        if run_id.startswith("p2_smoke"):
            continue  # P2 pilot smoke test, not part of the matrix
        workload = s["workload"]
        engine = s["engine"]
        frac, mm, raw = cache_stats(workload, run_id, engine)
        rows.append({
            "run_id": run_id,
            "workload": workload,
            "engine": engine,
            "rate": s["rate"],
            "ttft_p50": s["ttft_s"]["p50"],
            "ttft_p90": s["ttft_s"]["p90"],
            "ttft_p99": s["ttft_s"]["p99"],
            "itl_p50": s["itl_s"]["p50"],
            "tok_s": s["tokens_per_second"],
            "req_s": s["requests_per_second"],
            "isl_p50": s["isl_tokens"]["p50"],
            "completion_ratio": s["completion_ratio"],
            "growth": s.get("ttft_growth_ratio"),
            "tags": s["tags"],
            "saturated": "saturated" in s["tags"],
            "per_req_frac": s.get("measured_cached_fraction"),
            "counter_frac": frac,
            "mm_cache_frac": mm,
            "raw": raw,
        })
    json.dump(rows, open("scripts/_p4_data.json", "w"), indent=1)

    hdr = (f"{'run_id':<36}{'eng':<7}{'rate':>7}{'p50':>9}{'p90':>9}{'itl':>8}"
           f"{'tok/s':>8}{'per-req':>9}{'counter':>9}{'mm$':>7}{'grw':>6}  tag")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        pr = f"{r['per_req_frac']:.3f}" if r["per_req_frac"] is not None else "-"
        cf = f"{r['counter_frac']:.3f}" if r["counter_frac"] is not None else "-"
        mm = f"{r['mm_cache_frac']:.3f}" if r["mm_cache_frac"] is not None else "-"
        print(f"{r['run_id']:<36}{r['engine']:<7}{r['rate']:>7}{r['ttft_p50']:>9.3f}"
              f"{r['ttft_p90']:>9.3f}{r['itl_p50']:>8.4f}{r['tok_s']:>8.1f}"
              f"{pr:>9}{cf:>9}{mm:>7}{r['growth']:>6.2f}  "
              f"{'SAT' if r['saturated'] else 'ok'}")
    print(f"\n{len(rows)} runs")
    return rows


if __name__ == "__main__":
    main()
