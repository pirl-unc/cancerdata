#!/usr/bin/env python3
"""Generate the compact cancer-reference-expression source manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref._reference_sources import (
    TREEHOUSE_TCGA_LEGACY_COHORT,
    TREEHOUSE_TCGA_SAMPLES_COHORT,
    TREEHOUSE_TCGA_SARC_HISTOLOGY_CODES,
    TREEHOUSE_TCGA_SARC_HISTOLOGY_COHORT,
)
from oncoref.source_matrices import registry as source_matrix_registry

SOURCE_COLUMNS = [
    "cancer_code",
    "source_cohort",
    "source_project",
    "source_version",
    "tumor_origin",
    "metastasis_site",
    "processing_pipeline",
    "notes",
]
OUTPUT_COLUMNS = [
    *SOURCE_COLUMNS,
    "n_reference_genes",
    "n_reference_samples",
    "selected",
]


def _first_present(values: pd.Series):
    for value in values:
        if pd.notna(value) and str(value):
            return " ".join(str(value).split()) if isinstance(value, str) else value
    return pd.NA


def _canonicalize_source_labels(frame: pd.DataFrame) -> pd.DataFrame:
    sources = frame["source_cohort"]
    sarc_histology = frame["cancer_code"].isin(TREEHOUSE_TCGA_SARC_HISTOLOGY_CODES) & sources.isin(
        [TREEHOUSE_TCGA_LEGACY_COHORT, TREEHOUSE_TCGA_SAMPLES_COHORT]
    )
    generic_tcga = sources.eq(TREEHOUSE_TCGA_LEGACY_COHORT) & ~sarc_histology
    if not (sarc_histology.any() or generic_tcga.any()):
        return frame
    out = frame.copy()
    out.loc[sarc_histology, "source_cohort"] = TREEHOUSE_TCGA_SARC_HISTOLOGY_COHORT
    out.loc[generic_tcga, "source_cohort"] = TREEHOUSE_TCGA_SAMPLES_COHORT
    return out


def _shard_source_records(path: Path, *, chunksize: int) -> list[dict]:
    records: dict[tuple[str, str], dict] = {}
    gene_ids: dict[tuple[str, str], set[str]] = {}
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk = _canonicalize_source_labels(chunk)
        for (code, source), group in chunk.groupby(
            ["cancer_code", "source_cohort"], dropna=False, sort=False
        ):
            key = (str(code), str(source))
            if key not in records:
                records[key] = {
                    "cancer_code": key[0],
                    "source_cohort": key[1],
                    **{
                        column: _first_present(group[column]) if column in group.columns else pd.NA
                        for column in SOURCE_COLUMNS[2:]
                    },
                    "n_reference_samples": pd.NA,
                }
                gene_ids[key] = set()
            gene_ids[key].update(group["Ensembl_Gene_ID"].dropna().astype(str))
            if "n_samples" in group.columns:
                n_samples = pd.to_numeric(group["n_samples"], errors="coerce").max()
                current = records[key]["n_reference_samples"]
                if pd.notna(n_samples) and (pd.isna(current) or n_samples > current):
                    records[key]["n_reference_samples"] = n_samples

    for key, record in records.items():
        record["n_reference_genes"] = len(gene_ids[key])
    return list(records.values())


def _selected_sources() -> dict[str, str]:
    registry = source_matrix_registry()
    if registry["cancer_code"].astype(str).duplicated().any():
        raise ValueError("source-matrix registry contains duplicate cancer codes")
    return dict(
        zip(
            registry["cancer_code"].astype(str),
            registry["source_cohort"].astype(str),
        )
    )


def build_reference_availability(
    shard_dir: Path,
    *,
    chunksize: int = 100_000,
    selected_sources: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build one row per physical source and mark only registry-selected rows.

    Gene/sample counts describe availability; they never decide selection. The
    authoritative ``cancer_code -> source_cohort`` mapping comes from
    ``source-matrices.csv`` so adding a broader alternative source cannot silently
    promote it or pool it into the selected reference.
    """
    records = []
    seen = set()
    paths = sorted(list(shard_dir.glob("*.csv")) + list(shard_dir.glob("*.csv.gz")))
    if not paths:
        raise FileNotFoundError(f"no reference-expression CSV shards under {shard_dir}")
    for path in paths:
        for record in _shard_source_records(path, chunksize=chunksize):
            key = (record["cancer_code"], record["source_cohort"])
            if key in seen:
                raise ValueError(f"source identity {key!r} is split across multiple shards")
            seen.add(key)
            records.append(record)

    table = pd.DataFrame.from_records(records)
    selected_sources = _selected_sources() if selected_sources is None else selected_sources
    available_keys = set(zip(table["cancer_code"].astype(str), table["source_cohort"].astype(str)))
    missing = sorted(
        (code, source)
        for code, source in selected_sources.items()
        if code in set(table["cancer_code"].astype(str)) and (code, source) not in available_keys
    )
    if missing:
        raise ValueError(f"selected source rows are absent from summary shards: {missing}")
    unregistered_codes = sorted(set(table["cancer_code"].astype(str)) - set(selected_sources))
    if unregistered_codes:
        raise ValueError(
            "summary shards contain cancer codes absent from the source-matrix registry: "
            f"{unregistered_codes}"
        )
    table["selected"] = [
        selected_sources.get(str(code)) == str(source)
        for code, source in zip(table["cancer_code"], table["source_cohort"])
    ]
    origin_rank = (
        table["tumor_origin"]
        .fillna("")
        .astype(str)
        .str.lower()
        .map({"primary": 0, "mixed": 1, "metastasis": 2})
    )
    table["_origin_rank"] = origin_rank.fillna(99).astype(int)
    table = table.sort_values(
        [
            "cancer_code",
            "selected",
            "n_reference_genes",
            "n_reference_samples",
            "_origin_rank",
            "source_cohort",
        ],
        ascending=[True, False, False, False, True, True],
        kind="stable",
    )
    return table.drop(columns="_origin_rank").reset_index(drop=True)[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("oncoref/data/cancer-reference-expression-availability.csv"),
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    table = build_reference_availability(args.shard_dir, chunksize=args.chunksize)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"wrote {len(table)} source rows to {args.output}")


if __name__ == "__main__":
    main()
