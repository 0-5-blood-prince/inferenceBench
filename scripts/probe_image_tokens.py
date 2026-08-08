#!/usr/bin/env python3
"""P0 step 1: probe the model's per-image token count, per resolution.

Everything in workloads/build.py depends on this number, and it is not knowable
offline: dynamic-resolution ViTs make image tokens a function of resolution, and
each engine applies its own preprocessing. Run this against a live server before
building workloads.

    python3 scripts/probe_image_tokens.py --base-url http://localhost:8000 \
        --model <id> --resolutions 448x448,896x896,1344x1344

For each resolution it sends one request with the image and one without,
identical text, and reports the difference in prompt_tokens. Fixing one
resolution from this table is what holds T constant across W1-W3 (SPEC 5).
"""

import argparse
import base64
import io
import json
import random
import sys
import urllib.request

PROBE_TEXT = "Describe this image in one word."


def noise_png(width: int, height: int, seed: int = 0) -> bytes:
    from PIL import Image

    rng = random.Random(seed)
    raw = bytes(rng.getrandbits(8) for _ in range(width * height * 3))
    buf = io.BytesIO()
    Image.frombytes("RGB", (width, height), raw).save(buf, format="PNG",
                                                      compress_level=1)
    return buf.getvalue()


def chat(base_url: str, model: str, content, timeout: float) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--resolutions", default="448x448,896x896,1344x1344")
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()

    text_only = chat(args.base_url, args.model, PROBE_TEXT, args.timeout)
    baseline = text_only["usage"]["prompt_tokens"]
    print(f"text-only prompt_tokens: {baseline}")

    table = []
    for spec in args.resolutions.split(","):
        width, height = (int(v) for v in spec.strip().lower().split("x"))
        uri = "data:image/png;base64," + base64.b64encode(
            noise_png(width, height)).decode("ascii")
        content = [
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": PROBE_TEXT},
        ]
        try:
            resp = chat(args.base_url, args.model, content, args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"{spec}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        total = resp["usage"]["prompt_tokens"]
        table.append({"resolution": spec, "prompt_tokens": total,
                      "image_tokens": total - baseline})
        print(f"{spec}: prompt_tokens={total} image_tokens={total - baseline}")

    print("\nRecord the chosen row in the P0 log, then pass it to build.py:")
    print(json.dumps({"text_only_prompt_tokens": baseline, "probe": table}, indent=2))


if __name__ == "__main__":
    main()
