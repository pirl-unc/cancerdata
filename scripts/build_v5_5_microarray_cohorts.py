#!/usr/bin/env python
"""Build audited GEO microarray cohorts as explicitly tagged TPM proxies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.expression_engine import sample_columns
from oncoref.expression_source_adapters import (
    MICROARRAY_GROUPS,
    build_geo_microarray_group,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit source-specific output directory; valid with one --only group.",
    )
    parser.add_argument(
        "--only",
        choices=tuple(MICROARRAY_GROUPS),
        action="append",
        help="Build only the named physical source group. Defaults to all groups.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--ensembl-release", type=int, default=112)
    parser.add_argument("--high-expression-threshold", type=float, default=1.0)
    args = parser.parse_args(argv)

    selected = list(dict.fromkeys(args.only or MICROARRAY_GROUPS))
    if args.output_dir is not None and len(selected) != 1:
        parser.error("--output-dir requires exactly one --only group")
    cache_root = args.cache_root or Path.home() / ".cache" / "oncoref" / "expression"

    summary = {}
    for name in selected:
        group = MICROARRAY_GROUPS[name]
        result = build_geo_microarray_group(
            group,
            cache_dir=cache_root / group.source_id,
            output_dir=args.output_dir,
            force_download=args.force_download,
            ensembl_release=args.ensembl_release,
            high_expression_threshold=args.high_expression_threshold,
        )
        summary[name] = {
            "source_id": group.source_id,
            "matrix_paths": {code: str(path) for code, path in result.matrix_paths.items()},
            "sample_counts": {
                code: len(sample_columns(matrix)) for code, matrix in result.matrices.items()
            },
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
