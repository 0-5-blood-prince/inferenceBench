#!/usr/bin/env python3
"""Build the text-only control workloads (P4 "Text-only control", promoted
from P5's original priority-#1 stretch item after P5 was skipped as written
- see spec/phases/P5-stretch.md and spec/phases/P4-writeup.md).

Full reuse and Partial reuse, replicated with the image stripped out and the
removed image-token budget replaced by exact-token-count text padding inside
the shared head, so total T and every cacheable-fraction figure in
workloads/manifest.json stay identical to the multimodal twin. This isolates
whether the vLLM/SGLang TTFT gap is multimodal-prefill-specific (the
CUDA-graph-capture-asymmetry hypothesis) or general scheduling (the
cache-effect hypothesis) - the two competing primary hypotheses in P4's
mechanism bullet.

Reads the frozen workloads/manifest.json (does not recompute geometry - this
is a control built from the already-frozen constants, not a new design) and
each source workload's requests.jsonl, so it inherits the exact same
canonical-block / partial-reuse-offset structure per request, just with
[IMG shared] replaced by [PAD shared] of equal token length.

    python3 scripts/build_text_only.py --tokenizer google/gemma-4-31B-it
"""

import argparse
import json
import random
from pathlib import Path

SOURCE_TO_TEXT_SLUG = {
    "full-reuse": "full-reuse-text",
    "partial-reuse": "partial-reuse-text",
}

VOCAB = (
    "time person year way day thing man world life hand part child eye woman "
    "place work week case point government company number group problem fact be "
    "have do say get make go know take see come think look want give use find "
    "tell ask work seem feel try leave call good new first last long great "
    "little own other old right big high different small large next early young "
    "important few public bad same able the of and a in that it for not on with "
    "as you at this but his by from they we her she or an will my one all would "
    "there their what so up out about who which when make can like just him know "
    "into your over think also back after use two how our well even want because "
    "any these give most us water light sound field study story month night book "
    "line road river house school system program question during number without "
    "before between under against through above below across behind beyond"
).split()


def exact(tok, rng, n_tokens: int) -> str:
    if n_tokens <= 0:
        return ""
    text = " ".join(rng.choice(VOCAB) for _ in range(max(8, int(n_tokens * 1.5))))
    while len(tok.encode(text, add_special_tokens=False)) < n_tokens + 16:
        text += " " + " ".join(rng.choice(VOCAB) for _ in range(64))
    for _ in range(24):
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == n_tokens:
            return text
        if len(ids) > n_tokens:
            text = tok.decode(ids[:n_tokens]).rstrip()
        else:
            text += " " + " ".join(rng.choice(VOCAB) for _ in range(max(1, n_tokens - len(ids))))
    raise RuntimeError(f"could not hit exactly {n_tokens} tokens")


def strip_image(messages: list) -> str:
    """Pull the text part back out of a [image_url, text] user content list."""
    user = messages[-1]["content"]
    for part in user:
        if part.get("type") == "text":
            return part["text"]
    raise ValueError("no text part found in message content")


def build(args) -> None:
    from transformers import AutoTokenizer

    root = Path(args.workloads_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    image_tokens = manifest["geometry"]["image_tokens"]
    system_prompt = None

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)
    pad_text = exact(tok, rng, image_tokens)

    for src_slug, text_slug in SOURCE_TO_TEXT_SLUG.items():
        src_dir = root / src_slug
        rows = [json.loads(line) for line in (src_dir / "requests.jsonl").read_text().splitlines()]

        out_dir = root / text_slug
        (out_dir / "results").mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

        out_rows = []
        for row in rows:
            messages = row["messages"]
            if system_prompt is None:
                system_prompt = messages[0]["content"]
            body_text = strip_image(messages)
            new_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{pad_text} {body_text}".strip()},
            ]
            out_row = dict(row)
            out_row["messages"] = new_messages
            # cacheable-token accounting is unaffected: [SYS][PAD shared] takes
            # the image's place in the shared head 1:1, everything downstream
            # of it in the request is untouched.
            out_rows.append(out_row)

        with (out_dir / "requests.jsonl").open("w") as fh:
            for row in out_rows:
                fh.write(json.dumps(row) + "\n")

        print(f"{text_slug}: {len(out_rows)} requests, image_tokens={image_tokens} "
              f"replaced with {len(tok.encode(pad_text, add_special_tokens=False))} "
              f"pad tokens -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workloads-dir", default="workloads")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--seed", type=int, default=0)
    build(p.parse_args())


if __name__ == "__main__":
    main()
