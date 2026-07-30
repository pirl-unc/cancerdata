#!/usr/bin/env python
"""Build an SRA source from NCBI-generated gene-count tables.

This command never downloads raw reads and never runs an aligner or quantifier.
It verifies NCBI's processed count tables and a pinned RefSeq GFF, converts the
counts to gene TPM, and emits the standard oncoref source-matrix artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oncoref.expression_builders import (
    build_sra_ncbi_count_source_matrices,
    sra_ncbi_count_source_from_registry,
)


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
        help="Build cache. Defaults to ~/.cache/oncoref/expression/<source-id>/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Canonical matrices and sidecars. Defaults to cache_dir/derived.",
    )
    parser.add_argument(
        "--counts-dir",
        type=Path,
        default=None,
        help="Use reviewed <run>.ncbi-gene-counts.tsv files from this directory.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help="Use a reviewed RefSeq GFF; its registry SHA-256 is still verified.",
    )
    parser.add_argument(
        "--assembly-report",
        type=Path,
        default=None,
        help="Use a reviewed NCBI assembly report; its registry SHA-256 is still verified.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--high-expression-threshold", type=float, default=1.0)
    args = parser.parse_args(argv)

    source_id = args.source_id or args.source_id_arg
    if source_id is None:
        raise SystemExit("provide --source-id <id>")
    source = sra_ncbi_count_source_from_registry(source_id, registry_path=args.registry)
    cache_dir = args.cache_dir or Path.home() / ".cache" / "oncoref" / "expression" / source_id
    count_paths = None
    if args.counts_dir is not None:
        count_paths = {
            run.accession: args.counts_dir / f"{run.accession}.ncbi-gene-counts.tsv"
            for run in source.runs
        }

    result = build_sra_ncbi_count_source_matrices(
        source,
        cache_dir=cache_dir,
        output_dir=args.output_dir,
        count_paths=count_paths,
        annotation_path=args.annotation,
        assembly_report_path=args.assembly_report,
        force_download=args.force_download,
        high_expression_threshold=args.high_expression_threshold,
    )
    summary = {
        "source_id": source_id,
        "source_cohort": source.source_cohort,
        "matrix_paths": {code: str(path) for code, path in result.matrix_paths.items()},
        "sidecar_paths": {name: str(path) for name, path in result.sidecar_paths.items()},
        "sample_counts": {
            code: len(matrix.columns) - 2 for code, matrix in result.matrices.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
