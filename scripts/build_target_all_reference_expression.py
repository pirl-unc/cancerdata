#!/usr/bin/env python
"""Build TARGET B-ALL and T-ALL matrices with public lineage evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.expression_engine import sample_columns
from oncoref.expression_source_adapters import build_target_all_source_matrices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--file-dir",
        type=Path,
        default=None,
        help="Directory containing manifest source_file_name values for offline builds.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--high-expression-threshold", type=float, default=1.0)
    args = parser.parse_args(argv)

    cache = args.cache_dir or (Path.home() / ".cache" / "oncoref" / "expression" / "target-all")
    manifest = pd.read_csv(args.manifest) if args.manifest is not None else None
    file_paths = None
    if args.file_dir is not None:
        if manifest is None:
            parser.error("--file-dir requires --manifest")
        file_paths = {
            str(name): args.file_dir / str(name)
            for name in manifest["source_file_name"].astype(str)
        }
    result = build_target_all_source_matrices(
        cache_dir=cache,
        output_dir=args.output_dir,
        manifest=manifest,
        file_paths=file_paths,
        force_download=args.force_download,
        high_expression_threshold=args.high_expression_threshold,
    )
    print(
        json.dumps(
            {
                "source_id": "target-all",
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
