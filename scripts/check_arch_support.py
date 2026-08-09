#!/usr/bin/env python3
"""Does each engine recognise the model's architecture? (P0, before any download)

    <venv>/bin/python scripts/check_arch_support.py --arch Gemma4ForConditionalGeneration

Run once per engine venv, before pulling tens of GB of weights. P0's hard rule
is "do not debug engine support - switch models and move on", and that rule is
only cheap if it fires early. An unsupported architecture is not a bug to fix,
it is a model to replace.

If only one engine supports the architecture, the comparison is impossible: the
whole design rests on serving one identical model on both engines.

Exit 0 = supported, 1 = not supported, 2 = could not determine.
"""

import argparse
import importlib
import sys


def check_vllm(arch: str):
    try:
        from vllm.model_executor.models.registry import ModelRegistry
    except Exception as exc:  # noqa: BLE001
        return None, f"vllm registry import failed: {type(exc).__name__}: {exc}"
    for attr in ("get_supported_archs", "get_supported_architectures"):
        fn = getattr(ModelRegistry, attr, None)
        if callable(fn):
            try:
                archs = list(fn())
                return arch in archs, f"{len(archs)} architectures registered"
            except Exception:  # noqa: BLE001
                pass
    try:  # older/newer layouts keep the mapping on the instance
        archs = list(getattr(ModelRegistry, "models", {}) or {})
        if archs:
            return arch in archs, f"{len(archs)} architectures registered"
    except Exception:  # noqa: BLE001
        pass
    return None, "could not read vllm's architecture registry"


def check_sglang(arch: str):
    for mod, attr in (("sglang.srt.models.registry", "ModelRegistry"),
                      ("sglang.srt.model_loader.registry", "ModelRegistry")):
        try:
            registry = getattr(importlib.import_module(mod), attr)
        except Exception:  # noqa: BLE001
            continue
        for name in ("models", "_registry"):
            table = getattr(registry, name, None)
            if isinstance(table, dict) and table:
                return arch in table, f"{len(table)} architectures registered"
        fn = getattr(registry, "get_supported_archs", None)
        if callable(fn):
            try:
                archs = list(fn())
                return arch in archs, f"{len(archs)} architectures registered"
            except Exception:  # noqa: BLE001
                pass
    return None, "could not read sglang's architecture registry"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, help="e.g. Gemma4ForConditionalGeneration")
    p.add_argument("--engine", choices=("vllm", "sglang"),
                   help="defaults to whichever is importable in this venv")
    args = p.parse_args()

    engine = args.engine
    if not engine:
        for candidate in ("vllm", "sglang"):
            try:
                importlib.import_module(candidate)
                engine = candidate
                break
            except Exception:  # noqa: BLE001
                continue
    if not engine:
        print("FAIL: neither vllm nor sglang importable in this interpreter")
        sys.exit(2)

    version = "unknown"
    try:
        version = getattr(importlib.import_module(engine), "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass

    supported, detail = (check_vllm if engine == "vllm" else check_sglang)(args.arch)
    label = f"{engine} {version}"

    if supported is None:
        print(f"UNKNOWN  {label}: {detail}")
        print("Determine by hand before downloading weights - do not assume.")
        sys.exit(2)
    if supported:
        print(f"SUPPORTED  {label} recognises {args.arch}  ({detail})")
        sys.exit(0)
    print(f"NOT SUPPORTED  {label} does not recognise {args.arch}  ({detail})")
    print("P0 hard rule: switch models, do not debug engine support.")
    sys.exit(1)


if __name__ == "__main__":
    main()
