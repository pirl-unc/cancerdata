#!/usr/bin/env python
"""Build the checksum-pinned OpenPBTA v23 primary CRANIO reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.expression_engine import sample_columns
from oncoref.expression_source_adapters import build_openpbta_cranio_source_matrices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expression-path", type=Path, default=None)
    parser.add_argument("--histologies-path", type=Path, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--high-expression-threshold", type=float, default=1.0)
    args = parser.parse_args(argv)

    cache = args.cache_dir or (
        Path.home() / ".cache" / "oncoref" / "expression" / "openpbta-v23-cranio"
    )
    result = build_openpbta_cranio_source_matrices(
        cache_dir=cache,
        output_dir=args.output_dir,
        expression_path=args.expression_path,
        histologies_path=args.histologies_path,
        force_download=args.force_download,
        high_expression_threshold=args.high_expression_threshold,
    )
    print(
        json.dumps(
            {
                "source_id": "openpbta-v23-cranio",
                "matrix_paths": {code: str(path) for code, path in result.matrix_paths.items()},
                "sample_counts": {
                    code: len(sample_columns(matrix)) for code, matrix in result.matrices.items()
                },
                "sidecar_paths": {name: str(path) for name, path in result.sidecar_paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
