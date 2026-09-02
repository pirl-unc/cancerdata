# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Structured entity-level driver spectra.

Diagnosis and molecular status are separate axes. This table records driver
events observed across a cancer entity; it never assigns an event to an
individual sample. Sample-level evidence lives in :mod:`oncoref.samples`.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .cancer_types import resolve_cancer_type
from .gene_ids import canonical_gene_id
from .load_dataset import _register_derived_cache, get_data

DRIVER_EVIDENCE_SCOPE_KINDS = frozenset({"pan_cancer", "source_cancer_code"})
DRIVER_MIGRATION_STATUSES = frozenset({"migrated", "rejected", "awaiting_source"})


def _require_columns(frame: pd.DataFrame, dataset: str, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")


@lru_cache(maxsize=1)
def _driver_gene_evidence_frame() -> pd.DataFrame:
    dataset = "driver-gene-evidence"
    frame = get_data(dataset, copy=False)
    _require_columns(
        frame,
        dataset,
        {
            "record_id",
            "gene_symbol",
            "ensembl_gene_id",
            "scope_kind",
            "source_scope",
            "source_record_key",
            "pmid",
            "source_locator",
            "source_file_sha256",
        },
    )
    if frame["record_id"].duplicated().any():
        raise ValueError(f"{dataset} contains duplicate record_id values")
    if frame["source_record_key"].duplicated().any():
        raise ValueError(f"{dataset} contains duplicate source_record_key values")
    invalid_scopes = sorted(set(frame["scope_kind"].astype(str)) - DRIVER_EVIDENCE_SCOPE_KINDS)
    if invalid_scopes:
        raise ValueError(f"{dataset} contains invalid scope_kind values: {invalid_scopes}")
    for row in frame.itertuples(index=False):
        if canonical_gene_id(row.gene_symbol) != row.ensembl_gene_id:
            raise ValueError(
                f"{dataset} record {row.record_id} has a symbol/Ensembl mismatch: "
                f"{row.gene_symbol}/{row.ensembl_gene_id}"
            )
    return frame


@lru_cache(maxsize=1)
def _driver_variant_evidence_frame() -> pd.DataFrame:
    dataset = "driver-variant-evidence"
    frame = get_data(dataset, copy=False)
    _require_columns(
        frame,
        dataset,
        {
            "record_id",
            "gene_symbol",
            "ensembl_gene_id",
            "ensembl_transcript_id",
            "protein_change",
            "variant_notation",
            "scope_kind",
            "source_scope",
            "pmid",
            "source_locator",
            "source_file_sha256",
        },
    )
    if frame["record_id"].duplicated().any():
        raise ValueError(f"{dataset} contains duplicate record_id values")
    key = ["ensembl_gene_id", "ensembl_transcript_id", "protein_change"]
    if frame.duplicated(key).any():
        raise ValueError(f"{dataset} contains duplicate gene/transcript/protein-change keys")
    if not frame["ensembl_transcript_id"].astype(str).str.fullmatch(r"ENST\d+").all():
        raise ValueError(f"{dataset} contains a malformed Ensembl transcript ID")
    if not frame["protein_change"].astype(str).str.fullmatch(r"p\.[A-Z*][A-Za-z0-9_*?=]+").all():
        raise ValueError(f"{dataset} contains non-normalized protein-change notation")
    for row in frame.itertuples(index=False):
        if canonical_gene_id(row.gene_symbol) != row.ensembl_gene_id:
            raise ValueError(
                f"{dataset} record {row.record_id} has a symbol/Ensembl mismatch: "
                f"{row.gene_symbol}/{row.ensembl_gene_id}"
            )
    return frame


_register_derived_cache(_driver_gene_evidence_frame.cache_clear)
_register_derived_cache(_driver_variant_evidence_frame.cache_clear)


def driver_gene_evidence_df() -> pd.DataFrame:
    """Source-anchored cancer-driver gene evidence as a defensive copy.

    Rows preserve Bailey et al. Table S1's source cancer scope, Ensembl release
    92 gene identity, driver role, decision, and frequencies. ``scope_kind``
    distinguishes the source's pan-cancer rows from its cancer-code-scoped rows;
    it does not invent an oncoref cancer entity.
    """
    return _driver_gene_evidence_frame().copy()


def driver_variant_evidence_df() -> pd.DataFrame:
    """Source-anchored recurrent driver variants as a defensive copy.

    These are Bailey et al. Table S4's pan-cancer records. Each row preserves
    the Ensembl release 92 gene/transcript identifiers and normalized HGVS
    protein notation, without assigning a context-free variant to a cancer type.
    """
    return _driver_variant_evidence_frame().copy()


def _filter_gene(frame: pd.DataFrame, gene: str | None) -> pd.DataFrame:
    if gene is None:
        return frame
    ensembl_id = canonical_gene_id(gene)
    if ensembl_id is None:
        return frame.iloc[0:0]
    return frame.loc[frame["ensembl_gene_id"].eq(ensembl_id)]


def driver_gene_evidence(
    gene: str | None = None, *, source_scope: str | None = None
) -> pd.DataFrame:
    """Driver-gene evidence filtered by gene and/or source-native cancer scope."""
    frame = _filter_gene(_driver_gene_evidence_frame(), gene)
    if source_scope is not None:
        frame = frame.loc[
            frame["source_scope"].astype(str).str.upper().eq(str(source_scope).strip().upper())
        ]
    return frame.reset_index(drop=True).copy()


def driver_variant_evidence(
    gene: str | None = None, *, protein_change: str | None = None
) -> pd.DataFrame:
    """Pan-cancer recurrent-driver evidence filtered by gene/protein change."""
    frame = _filter_gene(_driver_variant_evidence_frame(), gene)
    if protein_change is not None:
        frame = frame.loc[frame["protein_change"].eq(str(protein_change).strip())]
    return frame.reset_index(drop=True).copy()


def driver_legacy_migration_audit_df(dataset: str | None = None) -> pd.DataFrame:
    """Disposition of every frozen driver row in its canonical evidence table.

    ``migration_status`` is one of ``migrated``, ``rejected``, or
    ``awaiting_source``. The current pinned snapshots are byte-identical to the
    documented OpenVax exports, so all rows have a publication-anchored
    destination. The audit is computed from the shipped legacy and canonical
    tables, making future drift visible instead of silently dropping a row.
    """
    accepted = {"cancer-driver-genes", "cancer-driver-variants"}
    if dataset is not None and dataset not in accepted:
        raise ValueError(f"dataset must be one of {sorted(accepted)}")

    specs = (
        (
            "cancer-driver-genes",
            "driver-gene-evidence",
            "bailey2018-s1-",
            "KEY",
            _driver_gene_evidence_frame(),
        ),
        (
            "cancer-driver-variants",
            "driver-variant-evidence",
            "bailey2018-s4-",
            None,
            _driver_variant_evidence_frame(),
        ),
    )
    audit_rows = []
    for legacy_name, destination_name, prefix, key_column, destination in specs:
        if dataset is not None and legacy_name != dataset:
            continue
        legacy = get_data(legacy_name, copy=False)
        destinations = destination.set_index("record_id", drop=False)
        for position, row in enumerate(legacy.itertuples(index=False), start=1):
            record_id = f"{prefix}{position:04d}"
            if key_column is not None:
                legacy_key = str(getattr(row, key_column))
            else:
                legacy_key = "|".join(
                    (
                        str(row.Ensembl_Gene_ID),
                        str(row.Ensembl_Transcript_ID),
                        str(row.Mutation),
                    )
                )
            migrated = record_id in destinations.index
            destination_row = destinations.loc[record_id] if migrated else None
            if migrated and legacy_name == "cancer-driver-genes":
                observed = (
                    str(destination_row["gene_symbol"]),
                    str(destination_row["ensembl_gene_id"]),
                    str(destination_row["source_scope"]),
                    str(destination_row["source_record_key"]),
                )
                expected = (
                    str(row.Symbol),
                    str(row.Ensembl_Gene_ID),
                    str(row.Cancer),
                    str(row.KEY),
                )
                if observed != expected:
                    raise ValueError(
                        f"{destination_name} record {record_id} does not match its legacy row"
                    )
            elif migrated:
                observed = (
                    str(destination_row["gene_symbol"]),
                    str(destination_row["ensembl_gene_id"]),
                    str(destination_row["ensembl_transcript_id"]),
                    str(destination_row["protein_change"]),
                )
                expected = (
                    str(row.Symbol),
                    str(row.Ensembl_Gene_ID),
                    str(row.Ensembl_Transcript_ID),
                    str(row.Mutation),
                )
                if observed != expected:
                    raise ValueError(
                        f"{destination_name} record {record_id} does not match its legacy row"
                    )
            audit_rows.append(
                {
                    "legacy_dataset": legacy_name,
                    "legacy_row_number": position,
                    "legacy_key": legacy_key,
                    "migration_status": "migrated" if migrated else "awaiting_source",
                    "destination_dataset": destination_name if migrated else "",
                    "destination_record_id": record_id if migrated else "",
                    "source": destination_row["pmid"] if migrated else "",
                    "source_locator": destination_row["source_locator"] if migrated else "",
                    "reason": (
                        "source-identical pinned row with publication locator"
                        if migrated
                        else "no source-anchored destination row"
                    ),
                }
            )
    return pd.DataFrame(audit_rows)


def driver_legacy_migration_summary() -> pd.DataFrame:
    """Per-table counts for all driver migration disposition states."""
    audit = driver_legacy_migration_audit_df()
    counts = (
        audit.groupby(["legacy_dataset", "migration_status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=sorted(DRIVER_MIGRATION_STATUSES), fill_value=0)
    )
    counts.insert(0, "total", counts.sum(axis=1))
    return counts.reset_index()


def cancer_driver_spectrum_df() -> pd.DataFrame:
    """All entity-driver relationships as a defensive dataframe copy.

    Each row is one observed event or explicitly unresolved group. Counts are
    study-level evidence and may overlap when two drivers co-occur in a case.
    """
    return get_data("cancer-entity-driver-spectrum").copy()


def cancer_driver_spectrum(cancer_type: str) -> pd.DataFrame:
    """Driver-spectrum rows for one alias-resolved cancer entity."""
    code = resolve_cancer_type(cancer_type)
    return (
        cancer_driver_spectrum_df()
        .loc[lambda df: df["cancer_code"].eq(code)]
        .reset_index(drop=True)
    )


def observed_driver_events(
    cancer_type: str, *, include_unresolved: bool = False
) -> tuple[str, ...]:
    """Distinct published driver events for an entity, preserving table order.

    ``include_unresolved=True`` also returns the explicit molecularly unresolved
    state. The result describes an entity spectrum, not a diagnosis requirement.
    """
    rows = cancer_driver_spectrum(cancer_type)
    if not include_unresolved:
        rows = rows.loc[~rows["driver_class"].eq("unresolved")]
    return tuple(dict.fromkeys(rows["driver_event"].dropna().astype(str)))
