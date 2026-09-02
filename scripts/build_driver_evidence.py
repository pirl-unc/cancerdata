#!/usr/bin/env python3
"""Build source-anchored driver evidence from the pinned OpenVax snapshots.

The two legacy CSVs are byte-identical to the files at ``UPSTREAM_REF``.  Their
upstream README identifies the gene rows as Bailey et al. Table S1 and the
variant rows as Table S4.  This builder preserves every row and identifier while
adding an explicit evidence scope and immutable publication/file provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path

UPSTREAM_REPOSITORY = "https://github.com/openvax/gene-lists"
UPSTREAM_REF = "d0a2f0bcbc59e612fde6108df6cb28d469e0f90e"
PUBLICATION_PMID = "PMID:29625053"
PUBLICATION_DOI = "DOI:10.1016/j.cell.2018.02.060"
IDENTIFIER_RELEASE = "Ensembl release 92"

SOURCE_PINS = {
    "cancer-driver-genes.csv": "dbfb76d906e218dab2d94e90dd51f8db41dbccf6c82da11311baa418ea738fdf",
    "cancer-driver-variants.csv": (
        "d568385b221f0e28f619e3358b42cec7449b6d434237516df871b7f8e9ed4549"
    ),
}

GENE_FIELDS = (
    "record_id",
    "gene_symbol",
    "aliases",
    "ensembl_gene_id",
    "scope_kind",
    "source_scope",
    "source_record_key",
    "driver_role",
    "decision",
    "tissue_frequency_percent",
    "pancancer_frequency_percent",
    "pmid",
    "doi",
    "source_locator",
    "identifier_release",
    "upstream_repository",
    "upstream_ref",
    "source_file_sha256",
)

VARIANT_FIELDS = (
    "record_id",
    "gene_symbol",
    "aliases",
    "ensembl_gene_id",
    "ensembl_transcript_id",
    "protein_change",
    "variant_notation",
    "scope_kind",
    "source_scope",
    "pmid",
    "doi",
    "source_locator",
    "identifier_release",
    "upstream_repository",
    "upstream_ref",
    "source_file_sha256",
)


def _verify_source(path: Path) -> None:
    expected = SOURCE_PINS[path.name]
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(f"{path}: sha256 {observed} does not match pinned {expected}")


def _percent(value: str) -> str:
    value = value.strip()
    return value.removesuffix("%") if value else ""


def _read_rows(path: Path) -> list[dict[str, str]]:
    _verify_source(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_gene_rows(source: Path) -> list[dict[str, str]]:
    rows = []
    for index, row in enumerate(_read_rows(source), start=1):
        source_scope = row["Cancer"].strip()
        rows.append(
            {
                "record_id": f"bailey2018-s1-{index:04d}",
                "gene_symbol": row["Symbol"].strip(),
                "aliases": row["Aliases"].strip(),
                "ensembl_gene_id": row["Ensembl_Gene_ID"].strip(),
                "scope_kind": ("pan_cancer" if source_scope == "PANCAN" else "source_cancer_code"),
                "source_scope": source_scope,
                "source_record_key": row["KEY"].strip(),
                "driver_role": row["Function"].strip(),
                "decision": row["Decision"].strip(),
                "tissue_frequency_percent": _percent(row["Tissue_Frequency"]),
                "pancancer_frequency_percent": _percent(row["Pancan_Frequency"]),
                "pmid": PUBLICATION_PMID,
                "doi": PUBLICATION_DOI,
                "source_locator": "Table S1",
                "identifier_release": IDENTIFIER_RELEASE,
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_ref": UPSTREAM_REF,
                "source_file_sha256": SOURCE_PINS[source.name],
            }
        )
    return rows


def build_variant_rows(source: Path) -> list[dict[str, str]]:
    rows = []
    for index, row in enumerate(_read_rows(source), start=1):
        protein_change = row["Mutation"].strip()
        if not protein_change.startswith("p."):
            raise ValueError(f"{source}: row {index} is not HGVS protein notation")
        rows.append(
            {
                "record_id": f"bailey2018-s4-{index:04d}",
                "gene_symbol": row["Symbol"].strip(),
                "aliases": row["Aliases"].strip(),
                "ensembl_gene_id": row["Ensembl_Gene_ID"].strip(),
                "ensembl_transcript_id": row["Ensembl_Transcript_ID"].strip(),
                "protein_change": protein_change,
                "variant_notation": "HGVS protein",
                "scope_kind": "pan_cancer",
                "source_scope": "TCGA PanCancer Atlas",
                "pmid": PUBLICATION_PMID,
                "doi": PUBLICATION_DOI,
                "source_locator": "Table S4",
                "identifier_release": IDENTIFIER_RELEASE,
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_ref": UPSTREAM_REF,
                "source_file_sha256": SOURCE_PINS[source.name],
            }
        )
    return rows


def _render(rows: list[dict[str, str]], fields: tuple[str, ...]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_or_check(path: Path, content: str, *, check: bool) -> None:
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"stale generated driver evidence: {path}")
        return
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("oncoref/data"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    genes = build_gene_rows(args.data_dir / "cancer-driver-genes.csv")
    variants = build_variant_rows(args.data_dir / "cancer-driver-variants.csv")
    _write_or_check(
        args.data_dir / "driver-gene-evidence.csv",
        _render(genes, GENE_FIELDS),
        check=args.check,
    )
    _write_or_check(
        args.data_dir / "driver-variant-evidence.csv",
        _render(variants, VARIANT_FIELDS),
        check=args.check,
    )
    print(f"driver evidence: {len(genes)} gene rows, {len(variants)} variant rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
