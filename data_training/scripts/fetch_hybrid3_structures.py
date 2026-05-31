#!/usr/bin/env python3
"""Fetch and parse HybriD3 atomic-structure records into an EWALD catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_training" / "src"))

from ewald_data_training.hybrid3 import (  # noqa: E402
    fetch_atomic_structure_datasets,
    load_fixture_records,
    download_structure_files,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--base-url", default="https://materials.hybrid3.duke.edu"
    )
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--fixture-root")
    args = parser.parse_args(argv)

    if args.fixture_root:
        records = load_fixture_records(args.fixture_root)
        if args.limit:
            records = records[: args.limit]
    else:
        records = fetch_atomic_structure_datasets(
            base_url=args.base_url,
            page_size=args.page_size,
            limit=args.limit,
            timeout=args.timeout,
        )
    summary = download_structure_files(
        records,
        output_root=args.output_root,
        base_url=args.base_url,
        timeout=args.timeout,
        fixture_root=args.fixture_root,
    )
    print(f"records={summary['records']}")
    print(f"ready={summary['ready']}")
    print(f"missing={summary['missing']}")
    print(f"catalog={summary['catalog_path']}")
    print(f"manifest={summary['manifest_path']}")
    return 0 if summary["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
