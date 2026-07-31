#!/usr/bin/env python
"""Build a generic registry-backed GEO-style expression source matrix.

This script is intentionally thin: source parsing, unit conversion, gene mapping,
parse diagnostics, and sample QC live in :mod:`oncoref.expression_builders`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.expression_builders import build_source_matrices, geo_matrix_source_from_registry


def _load_gene_lengths_kb(path: str | Path | None) -> pd.Series | None:
    if path is None:
        return None
    df = pd.read_csv(Path(path))
    if df.empty:
        raise SystemExit(f"gene-length table is empty: {path}")
    by_lower = {str(col).lower(): col for col in df.columns}
    id_col = (
        by_lower.get("ensembl_gene_id")
        or by_lower.get("gene_id")
        or by_lower.get("source_row_id")
        or df.columns[0]
    )
    if len(df.columns) < 2 and not any(
        key in by_lower for key in ("length_kb", "gene_length_kb", "length")
    ):
        raise SystemExit(f"gene-length table must contain an id column and a length column: {path}")
    length_col = (
        by_lower.get("length_kb")
        or by_lower.get("gene_length_kb")
        or by_lower.get("length")
        or df.columns[1]
    )
    lengths = pd.to_numeric(df[length_col], errors="coerce")
    out = pd.Series(lengths.to_numpy(dtype=float), index=df[id_col].astype(str).to_numpy())
    out = out[out.notna() & (out > 0)]
    if out.empty:
        raise SystemExit(f"gene-length table has no positive numeric lengths: {path}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_id_arg", nargs="?", help="Source id from expression_sources.yaml")
    parser.add_argument("--source-id", help="Source id from expression_sources.yaml")
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Optional expression_sources.yaml path; defaults to the packaged registry.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Source-file cache directory. Defaults to ~/.cache/oncoref/expression/<source-id>/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-code parquet matrices and sidecars. Defaults to cache_dir/derived.",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=None,
        help="Use an already-downloaded source matrix instead of registry file_url/cache lookup.",
    )
    parser.add_argument(
        "--gene-lengths-kb",
        type=Path,
        default=None,
        help=(
            "Optional CSV mapping source row ids to positive gene lengths in kb. "
            "Raw-count builds otherwise derive Ensembl gene lengths with pyensembl."
        ),
    )
    parser.add_argument("--ensembl-release", type=int, default=112)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--high-expression-threshold", type=float, default=1.0)
    args = parser.parse_args(argv)

    source_id = args.source_id or args.source_id_arg
    if source_id is None:
        raise SystemExit("provide --source-id <id>")

    source = geo_matrix_source_from_registry(source_id, registry_path=args.registry)
    cache_dir = args.cache_dir or Path.home() / ".cache" / "oncoref" / "expression" / source_id
    result = build_source_matrices(
        source,
        cache_dir=cache_dir,
        output_dir=args.output_dir,
        source_path=args.source_path,
        force_download=args.force_download,
        gene_lengths_kb=_load_gene_lengths_kb(args.gene_lengths_kb),
        ensembl_release=args.ensembl_release,
        high_expression_threshold=args.high_expression_threshold,
    )
    summary = {
        "source_id": source_id,
        "source_cohort": source.source_cohort,
        "matrix_paths": {code: str(path) for code, path in result.matrix_paths.items()},
        "sidecar_paths": {name: str(path) for name, path in result.sidecar_paths.items()},
        "sample_counts": {
            code: len([c for c in matrix.columns if c not in {"Ensembl_Gene_ID", "Symbol"}])
            for code, matrix in result.matrices.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
