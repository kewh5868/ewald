#!/usr/bin/env python3
"""Validate an EWALD training-data JSONL manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_training" / "src"))

from ewald_data_training.manifests import (  # noqa: E402
    read_jsonl_manifest,
    validate_manifest_files,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    samples = read_jsonl_manifest(args.manifest)
    errors = validate_manifest_files(samples, root=args.root)
    for error in errors:
        print(error)
    print(f"{len(samples)} samples, {len(errors)} missing file errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
