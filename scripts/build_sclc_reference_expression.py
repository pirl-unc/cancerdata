#!/usr/bin/env python
"""Build SCLC UCologne and its four marker-dominance subtype matrices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.expression_builders import (
    build_sclc_source_matrices,
    geo_matrix_source_from_registry,
)
from oncoref.expression_engine import sample_columns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", default="sclc-ucologne-2015")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-path", type=Path, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--high-expression-threshold", type=float, default=1.0)
    args = parser.parse_args(argv)

    source = geo_matrix_source_from_registry(args.source_id, registry_path=args.registry)
    cache_dir = args.cache_dir or (
        Path.home() / ".cache" / "oncoref" / "expression" / args.source_id
    )
    result = build_sclc_source_matrices(
        source,
        cache_dir=cache_dir,
        output_dir=args.output_dir,
        source_path=args.source_path,
        force_download=args.force_download,
        high_expression_threshold=args.high_expression_threshold,
    )
    print(
        json.dumps(
            {
                "source_id": args.source_id,
                "source_cohort": source.source_cohort,
                "matrix_paths": {
                    code: str(path) for code, path in result.matrix_paths.items()
                },
                "sample_counts": {
                    code: len(sample_columns(matrix))
                    for code, matrix in result.matrices.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
