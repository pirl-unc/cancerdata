#!/usr/bin/env python
"""Build a registry-backed raw-read SRA source with Salmon.

The source registry owns the complete run inventory, read checksums, tumor/control
roles, and pinned Ensembl transcriptome. This command downloads and quantifies all
declared runs, then emits the standard oncoref source-matrix artifacts while routing
only tumor runs into cancer reference output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oncoref.expression_builders import (
    build_sra_salmon_source_matrices,
    sra_salmon_source_from_registry,
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
        "--quant-dir",
        type=Path,
        default=None,
        help="Reuse reviewed Salmon outputs laid out as <run>/quant.sf.",
    )
    parser.add_argument(
        "--transcriptome",
        type=Path,
        default=None,
        help="Reuse the pinned transcriptome FASTA; its SHA-256 is still verified.",
    )
    parser.add_argument("--salmon-executable", default="salmon")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--force-quant", action="store_true")
    parser.add_argument("--high-expression-threshold", type=float, default=1.0)
    args = parser.parse_args(argv)

    if args.quant_dir is not None and (args.force_index or args.force_quant):
        parser.error("--quant-dir cannot be combined with --force-index or --force-quant")

    source_id = args.source_id or args.source_id_arg
    if source_id is None:
        raise SystemExit("provide --source-id <id>")
    source = sra_salmon_source_from_registry(source_id, registry_path=args.registry)
    cache_dir = args.cache_dir or Path.home() / ".cache" / "oncoref" / "expression" / source_id
    quant_paths = None
    if args.quant_dir is not None:
        quant_paths = {
            run.accession: args.quant_dir / run.accession / "quant.sf" for run in source.runs
        }

    result = build_sra_salmon_source_matrices(
        source,
        cache_dir=cache_dir,
        output_dir=args.output_dir,
        quant_paths=quant_paths,
        transcriptome_path=args.transcriptome,
        salmon_executable=args.salmon_executable,
        threads=args.threads,
        force_download=args.force_download,
        force_index=args.force_index,
        force_quant=args.force_quant,
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
