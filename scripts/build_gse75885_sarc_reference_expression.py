#!/usr/bin/env python
"""Build the three GSE75885 sarcoma source matrices from public GEO files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.expression_engine import sample_columns
from oncoref.expression_source_adapters import build_gse75885_source_matrices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expression-path", type=Path, default=None)
    parser.add_argument("--series-matrix-path", type=Path, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--high-expression-threshold", type=float, default=1.0)
    args = parser.parse_args(argv)

    cache = args.cache_dir or (Path.home() / ".cache" / "oncoref" / "expression" / "gse75885-sarc")
    result = build_gse75885_source_matrices(
        cache_dir=cache,
        output_dir=args.output_dir,
        expression_path=args.expression_path,
        series_matrix_path=args.series_matrix_path,
        force_download=args.force_download,
        high_expression_threshold=args.high_expression_threshold,
    )
    print(
        json.dumps(
            {
                "source_id": "gse75885-sarc",
                "matrix_paths": {code: str(path) for code, path in result.matrix_paths.items()},
                "sample_counts": {
                    code: len(sample_columns(matrix)) for code, matrix in result.matrices.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
