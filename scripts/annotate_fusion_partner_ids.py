#!/usr/bin/env python3
"""Add canonical partner identity and locus-kind columns to cancer fusions."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from oncoref.gene_ids import canonical_gene_id

IMMUNOGLOBULIN_LOCI = frozenset({"IGH", "IGK", "IGL"})
TCR_LOCI = frozenset({"TRA", "TRB", "TRD", "TRG"})


def _partner_identity(value: str) -> tuple[str, str]:
    symbol = value.strip().upper()
    if not symbol:
        return "", "none"
    if symbol in IMMUNOGLOBULIN_LOCI:
        return "", "immunoglobulin_locus"
    if symbol in TCR_LOCI:
        return "", "tcr_locus"
    ensembl_id = canonical_gene_id(symbol)
    if ensembl_id is None:
        raise ValueError(f"unresolved fusion partner gene: {symbol}")
    return ensembl_id, "gene"


def build(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        old_fields = [
            field
            for field in (reader.fieldnames or [])
            if field
            not in {
                "gene_5prime_ensembl_id",
                "gene_5prime_kind",
                "gene_3prime_ensembl_id",
                "gene_3prime_kind",
            }
        ]

    fields = []
    for field in old_fields:
        fields.append(field)
        if field == "gene_5prime":
            fields.extend(("gene_5prime_ensembl_id", "gene_5prime_kind"))
        elif field == "gene_3prime":
            fields.extend(("gene_3prime_ensembl_id", "gene_3prime_kind"))

    seen = set()
    for index, row in enumerate(rows, start=2):
        for side in ("5prime", "3prime"):
            ensembl_id, kind = _partner_identity(row.get(f"gene_{side}", ""))
            row[f"gene_{side}_ensembl_id"] = ensembl_id
            row[f"gene_{side}_kind"] = kind
        key = (row["cancer_code"], row["gene_5prime"], row["gene_3prime"])
        if key in seen:
            raise ValueError(f"duplicate logical fusion key at CSV row {index}: {key}")
        seen.add(key)

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("oncoref/data/cancer-fusions.csv"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = build(args.path)
    if args.check:
        if args.path.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"fusion partner annotations are stale: {args.path}")
    else:
        args.path.write_text(rendered, encoding="utf-8")
    print(f"fusion partner identities: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
