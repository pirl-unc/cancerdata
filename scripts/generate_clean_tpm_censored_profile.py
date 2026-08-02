#!/usr/bin/env python
"""Build the global clean-TPM censored-gene profile from clean PolyA data.

The existing curated censored table supplies the biology-defined ribosomal and
technical membership. Canonical protocol-sensitive structural RNA biotypes are
added to that same global list. Per-gene reference TPM is the median across the
Treehouse Tumor Compendium 25.01 PolyA log2(TPM+1) matrix.

The output is the single runtime contract for membership, compartment, and
within-compartment reference composition. The legacy symbol-only table is a
generated compatibility projection, never a second source of truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.gene_families import (
    CLEAN_TPM_CENSORED_REFERENCE_PROFILE_SOURCE,
    CLEAN_TPM_CENSORED_REFERENCE_PROFILE_VERSION,
    CLEAN_TPM_PROTOCOL_SENSITIVE_BIOTYPES,
    TECHNICAL_RNA_FAMILIES,
    gene_family,
)
from oncoref.gene_ids import canonical_gene_space

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = _ROOT / "oncoref" / "data" / "clean-tpm-censored-genes.csv"
_DEFAULT_LEGACY_OUTPUT = _ROOT / "oncoref" / "data" / "censored-gene-reference-tpm.csv"
_BASE_COLUMNS = ("Ensembl_Gene_ID", "Symbol", "category")


def global_censored_membership(path: Path) -> pd.DataFrame:
    """Return curated membership plus every canonical protocol-sensitive RNA."""
    curated = pd.read_csv(path, usecols=list(_BASE_COLUMNS), dtype=str)
    historical = pd.concat(
        [gene_family(name)[["Ensembl_Gene_ID", "Symbol"]] for name in TECHNICAL_RNA_FAMILIES],
        ignore_index=True,
    )
    historical["category"] = "technical"
    genes = canonical_gene_space()
    structural = genes.loc[
        genes["biotype"].astype(str).isin(CLEAN_TPM_PROTOCOL_SENSITIVE_BIOTYPES),
        ["ensembl_gene_id", "symbol"],
    ].rename(columns={"ensembl_gene_id": "Ensembl_Gene_ID", "symbol": "Symbol"})
    structural["category"] = "technical"
    combined = pd.concat([curated, historical, structural], ignore_index=True)
    combined = combined.drop_duplicates("Ensembl_Gene_ID", keep="first")
    order = combined["category"].map({"ribosomal_protein": 0, "technical": 1})
    if order.isna().any():
        invalid = sorted(combined.loc[order.isna(), "category"].unique())
        raise ValueError(f"unsupported clean-TPM categories: {invalid}")
    return (
        combined.assign(_category_order=order)
        .sort_values(["_category_order", "Symbol", "Ensembl_Gene_ID"], kind="stable")
        .drop(columns="_category_order")
        .reset_index(drop=True)
    )


def treehouse_polya_symbol_medians(path: Path, symbols: set[str]) -> pd.Series:
    """Linear-TPM median by requested symbol from a Treehouse PolyA parquet."""
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(filter=ds.field("Symbol").isin(sorted(symbols)))
    frame = table.to_pandas()
    if "Symbol" not in frame or len(frame.columns) < 2:
        raise ValueError("Treehouse PolyA matrix must contain Symbol and sample columns")
    if frame.empty:
        raise ValueError("none of the censored symbols occur in the Treehouse PolyA matrix")

    samples = [column for column in frame.columns if column != "Symbol"]
    log_values = frame[samples].apply(pd.to_numeric, errors="coerce")
    if log_values.isna().any().any():
        raise ValueError("Treehouse PolyA matrix contains missing/non-numeric requested values")
    linear = np.exp2(log_values) - 1.0
    linear.insert(0, "Symbol", frame["Symbol"].astype(str).to_numpy())
    by_symbol = linear.groupby("Symbol", sort=False)[samples].sum(min_count=1)
    return by_symbol.median(axis=1)


def build_profile(membership_path: Path, treehouse_polya_path: Path) -> pd.DataFrame:
    """Return the complete assay-independent censored-gene reference table."""
    membership = global_censored_membership(membership_path)
    medians = treehouse_polya_symbol_medians(
        treehouse_polya_path,
        set(membership["Symbol"].dropna().astype(str)),
    )
    out = membership.copy()
    out["reference_tpm"] = out["Symbol"].map(medians).fillna(0.0).clip(lower=0).round(6)
    out["reference_source"] = CLEAN_TPM_CENSORED_REFERENCE_PROFILE_SOURCE
    out["reference_profile_version"] = CLEAN_TPM_CENSORED_REFERENCE_PROFILE_VERSION
    return out


def _write_profile(profile: pd.DataFrame, output: Path, legacy_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output, index=False, lineterminator="\n")
    legacy = (
        profile[["Symbol", "reference_tpm"]]
        .drop_duplicates("Symbol", keep="first")
        .sort_values("Symbol", kind="stable")
    )
    legacy.to_csv(legacy_output, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treehouse-polya", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--legacy-output", type=Path, default=_DEFAULT_LEGACY_OUTPUT)
    args = parser.parse_args(argv)

    profile = build_profile(args.output, args.treehouse_polya)
    _write_profile(profile, args.output, args.legacy_output)
    print(
        f"wrote {len(profile)} censored genes "
        f"({int((profile['reference_tpm'] > 0).sum())} with positive PolyA reference TPM)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
