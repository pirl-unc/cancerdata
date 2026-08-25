# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loaders over the Human Protein Atlas normal-tissue reference data.

These fetch (once, cached) and parse the HPA tables registered in
:mod:`oncoref.reference_data`:

- ``hpa_rna_consensus`` — per-tissue RNA nTPM (``Gene``, ``Gene name``,
  ``Tissue``, ``nTPM``).
- ``hpa_normal_tissue`` — IHC protein detection (``Gene``, ``Gene name``,
  ``Tissue``, ``Cell type``, ``Level``, ``Reliability``).
- ``hpa_single_cell`` — single-cell-type RNA nTPM (``Gene``, ``Gene name``,
  ``Cell type``, ``nTPM``).

This is the normal-tissue expression evidence behind the cancer-testis-antigen
tissue-restriction definition and protein-level / single-cell comparisons.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import cache, lru_cache

import pandas as pd

from . import reference_data
from .cta_tissues import SAFETY_TISSUE_GROUPS
from .load_dataset import _register_derived_cache, get_data

SAFETY_TISSUE_MAPPING_KINDS: frozenset[str] = frozenset(
    {"exact", "reviewed_equivalent", "reviewed_substructure", "unavailable"}
)
SAFETY_TISSUE_COVERAGE_LEVELS: frozenset[str] = frozenset({"complete", "partial", "unavailable"})
SAFETY_TISSUE_COVERAGE_STATES: frozenset[str] = frozenset({"complete", "partial", "unavailable"})

_SAFETY_TISSUE_MAPPING_COLUMNS = (
    "source_name",
    "source_version",
    "safety_group",
    "requested_tissue",
    "source_tissue",
    "mapping_kind",
    "coverage_level",
    "mapping_reference",
    "notes",
)


class SafetyTissueResolutionError(ValueError):
    """A safety-tissue group cannot be resolved under the requested contract."""


@dataclass(frozen=True)
class SafetyTissueMapping:
    """One conceptual safety tissue mapped to source-native tissue labels."""

    requested_tissue: str
    source_tissues: tuple[str, ...]
    mapping_kind: str
    coverage_level: str
    mapping_references: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def available(self) -> bool:
        """Whether at least one source-native tissue represents this concept."""
        return bool(self.source_tissues)

    @property
    def complete(self) -> bool:
        """Whether the source fully represents this conceptual tissue."""
        return self.coverage_level == "complete"


@dataclass(frozen=True)
class SafetyTissueResolution:
    """Deterministic source-aware resolution of a safety-tissue group."""

    safety_group: str
    source_name: str
    source_version: str
    source_url: str
    mappings: tuple[SafetyTissueMapping, ...]
    coverage_state: str

    @property
    def requested_tissues(self) -> tuple[str, ...]:
        """Conceptual tissues configured for the safety group."""
        return tuple(mapping.requested_tissue for mapping in self.mappings)

    @property
    def source_tissues(self) -> tuple[str, ...]:
        """Deduplicated source-native labels safe to use for exact matching."""
        return tuple(
            sorted({tissue for mapping in self.mappings for tissue in mapping.source_tissues})
        )

    @property
    def fully_covered_tissues(self) -> tuple[str, ...]:
        """Conceptual tissues with complete source coverage."""
        return tuple(
            mapping.requested_tissue
            for mapping in self.mappings
            if mapping.coverage_level == "complete"
        )

    @property
    def partially_covered_tissues(self) -> tuple[str, ...]:
        """Conceptual tissues represented only by reviewed substructures."""
        return tuple(
            mapping.requested_tissue
            for mapping in self.mappings
            if mapping.coverage_level == "partial"
        )

    @property
    def unavailable_tissues(self) -> tuple[str, ...]:
        """Conceptual tissues with no source-native label in this source version."""
        return tuple(
            mapping.requested_tissue
            for mapping in self.mappings
            if mapping.coverage_level == "unavailable"
        )

    @property
    def complete(self) -> bool:
        """Whether every requested conceptual tissue is completely covered."""
        return self.coverage_state == "complete"


def _read_hpa(name: str, version: str | None = None) -> pd.DataFrame:
    """Load an HPA table, caching a columnar **parquet** copy next to the raw TSV
    so repeated/cold reads are fast and compact (HPA TSVs are large — single-cell
    is tens of MB; parquet is a few-fold smaller and skips re-parsing). The parquet
    cache is best-effort and regenerated if the TSV is re-downloaded."""
    tsv = reference_data.ensure(name, version)
    parquet = tsv.with_suffix(".parquet")
    if parquet.exists() and parquet.stat().st_mtime >= tsv.stat().st_mtime:
        return pd.read_parquet(parquet)
    df = pd.read_csv(tsv, sep="\t")
    # parquet cache is an optimization, not required — never fail a read over it.
    with contextlib.suppress(Exception):
        df.to_parquet(parquet, index=False)
    return df


@lru_cache(maxsize=1)
def hpa_rna_consensus() -> pd.DataFrame:
    """HPA RNA consensus per-tissue nTPM (downloads HPA v23 on first use)."""
    return _read_hpa("hpa_rna_consensus")


@cache
def _hpa_normal_tissue_for_version(version: str) -> pd.DataFrame:
    """Cache one HPA IHC table under its concrete source-version key."""
    return _read_hpa("hpa_normal_tissue", version)


def hpa_normal_tissue(version: str | None = None) -> pd.DataFrame:
    """HPA IHC protein detection per tissue/cell type for one source version."""
    concrete_version = reference_data.resolve_version("hpa_normal_tissue", version)
    return _hpa_normal_tissue_for_version(concrete_version)


@lru_cache(maxsize=1)
def hpa_single_cell() -> pd.DataFrame:
    """HPA single-cell-type RNA nTPM (downloads on first use)."""
    return _read_hpa("hpa_single_cell")


@lru_cache(maxsize=1)
def hpa_cell_type_expression() -> pd.DataFrame:
    """Wide per-gene HPA single-cell-type nTPM matrix.

    Returns ``Ensembl_Gene_ID``, ``Symbol``, then one numeric column per HPA
    single-cell type. This is the analysis-facing companion to the raw long-form
    :func:`hpa_single_cell` table.
    """
    df = hpa_single_cell()
    if df.empty:
        return pd.DataFrame(columns=["Ensembl_Gene_ID", "Symbol"])
    wide = df.pivot_table(
        index=["Gene", "Gene name"],
        columns="Cell type",
        values="nTPM",
        aggfunc="sum",
        fill_value=0.0,
    )
    wide = wide.reset_index().rename(columns={"Gene": "Ensembl_Gene_ID", "Gene name": "Symbol"})
    wide.columns.name = None
    return wide


@lru_cache(maxsize=1)
def _safety_tissue_mapping_table() -> pd.DataFrame:
    table = get_data("hpa-safety-tissue-map", copy=False).fillna("")
    missing = set(_SAFETY_TISSUE_MAPPING_COLUMNS) - set(table.columns)
    if missing:
        raise SafetyTissueResolutionError(
            f"HPA safety-tissue mapping table is missing columns: {sorted(missing)}"
        )
    table = table.loc[:, _SAFETY_TISSUE_MAPPING_COLUMNS].astype(str)

    invalid_kinds = set(table["mapping_kind"]) - SAFETY_TISSUE_MAPPING_KINDS
    invalid_levels = set(table["coverage_level"]) - SAFETY_TISSUE_COVERAGE_LEVELS
    if invalid_kinds or invalid_levels:
        raise SafetyTissueResolutionError(
            "invalid HPA safety-tissue mapping vocabulary: "
            f"mapping_kind={sorted(invalid_kinds)}, coverage_level={sorted(invalid_levels)}"
        )

    duplicate_rows = table.duplicated(
        ["source_name", "source_version", "safety_group", "requested_tissue", "source_tissue"]
    )
    if duplicate_rows.any():
        raise SafetyTissueResolutionError("duplicate HPA safety-tissue mapping rows")

    unavailable = table["coverage_level"] == "unavailable"
    invalid_availability = (unavailable & table["source_tissue"].ne("")) | (
        ~unavailable & table["source_tissue"].eq("")
    )
    if invalid_availability.any():
        raise SafetyTissueResolutionError(
            "unavailable mappings must omit source_tissue and available mappings must provide it"
        )

    for (source_name, source_version), source_rows in table.groupby(
        ["source_name", "source_version"], sort=True
    ):
        mapped_groups = set(source_rows["safety_group"])
        configured_groups = set(SAFETY_TISSUE_GROUPS)
        if mapped_groups != configured_groups:
            raise SafetyTissueResolutionError(
                f"{source_name!r} {source_version!r} maps safety groups "
                f"{sorted(mapped_groups)}, expected {sorted(configured_groups)}"
            )
        for group, configured_tissues in SAFETY_TISSUE_GROUPS.items():
            mapped_tissues = set(
                source_rows.loc[source_rows["safety_group"] == group, "requested_tissue"]
            )
            if mapped_tissues != configured_tissues:
                raise SafetyTissueResolutionError(
                    f"{source_name!r} {source_version!r} group {group!r} maps "
                    f"{sorted(mapped_tissues)}, expected {sorted(configured_tissues)}"
                )
    return table


_register_derived_cache(_safety_tissue_mapping_table.cache_clear)


def safety_tissue_mapping_table() -> pd.DataFrame:
    """Reviewed conceptual-to-source-native safety-tissue mappings.

    The returned table is a defensive copy. Use :func:`resolve_safety_tissue_group`
    for a validated, immutable selection contract.
    """
    return _safety_tissue_mapping_table().copy()


def resolve_safety_tissue_group(
    safety_group: str,
    *,
    source_name: str = "hpa_normal_tissue",
    source_version: str | None = None,
    require_complete: bool = True,
) -> SafetyTissueResolution:
    """Resolve a conceptual safety group to exact source-native tissue labels.

    Resolution is pinned to a concrete ``(source_name, source_version)`` and
    includes reviewed aliases/substructures and unavailable conceptual tissues.
    By default incomplete coverage raises instead of silently weakening a safety
    filter. Set ``require_complete=False`` only when the returned partial and
    unavailable coverage has been explicitly reviewed by the caller.
    """
    group = str(safety_group).strip().casefold()
    if group not in SAFETY_TISSUE_GROUPS:
        raise SafetyTissueResolutionError(
            f"unknown safety tissue group {safety_group!r}; "
            f"available: {', '.join(sorted(SAFETY_TISSUE_GROUPS))}"
        )
    try:
        version = reference_data.resolve_version(source_name, source_version)
    except reference_data.ReferenceDataError as exc:
        raise SafetyTissueResolutionError(str(exc)) from None

    table = _safety_tissue_mapping_table()
    rows = table[
        (table["source_name"] == source_name)
        & (table["source_version"] == version)
        & (table["safety_group"] == group)
    ]
    if rows.empty:
        supported = sorted(
            {
                (str(name), str(source_version))
                for name, source_version in zip(table["source_name"], table["source_version"])
            }
        )
        supported_text = ", ".join(f"{name}@{ver}" for name, ver in supported)
        raise SafetyTissueResolutionError(
            f"no safety-tissue mapping for {source_name!r} {version!r}; supported: {supported_text}"
        )

    mappings = []
    for requested_tissue, tissue_rows in rows.groupby("requested_tissue", sort=True):
        kinds = set(tissue_rows["mapping_kind"])
        levels = set(tissue_rows["coverage_level"])
        if len(kinds) != 1 or len(levels) != 1:
            raise SafetyTissueResolutionError(
                f"inconsistent mapping rows for {group!r}/{requested_tissue!r}"
            )
        mappings.append(
            SafetyTissueMapping(
                requested_tissue=str(requested_tissue),
                source_tissues=tuple(sorted(t for t in tissue_rows["source_tissue"] if t)),
                mapping_kind=next(iter(kinds)),
                coverage_level=next(iter(levels)),
                mapping_references=tuple(
                    sorted({ref for ref in tissue_rows["mapping_reference"] if ref})
                ),
                notes=tuple(sorted({note for note in tissue_rows["notes"] if note})),
            )
        )

    coverage_levels = {mapping.coverage_level for mapping in mappings}
    if coverage_levels == {"complete"}:
        coverage_state = "complete"
    elif coverage_levels == {"unavailable"}:
        coverage_state = "unavailable"
    else:
        coverage_state = "partial"
    if coverage_state not in SAFETY_TISSUE_COVERAGE_STATES:
        raise SafetyTissueResolutionError(f"invalid coverage state {coverage_state!r}")

    resolution = SafetyTissueResolution(
        safety_group=group,
        source_name=source_name,
        source_version=version,
        source_url=reference_data.provenance(source_name, version)["url"],
        mappings=tuple(mappings),
        coverage_state=coverage_state,
    )
    if require_complete and not resolution.complete:
        raise SafetyTissueResolutionError(
            f"safety tissue group {group!r} is {coverage_state} for "
            f"{source_name}@{version}; partial={list(resolution.partially_covered_tissues)}, "
            f"unavailable={list(resolution.unavailable_tissues)}. Pass "
            "require_complete=False only after reviewing this coverage."
        )
    return resolution


@cache
def _hpa_normal_tissue_labels_for_version(version: str) -> tuple[str, ...]:
    labels = _hpa_normal_tissue_for_version(version)["Tissue"].dropna().astype(str)
    return tuple(sorted({label.strip() for label in labels if label.strip()}))


def hpa_normal_tissue_labels(version: str | None = None) -> tuple[str, ...]:
    """Exact HPA IHC tissue labels available in one concrete source version."""
    concrete_version = reference_data.resolve_version("hpa_normal_tissue", version)
    return _hpa_normal_tissue_labels_for_version(concrete_version)


def resolve_hpa_normal_tissue_label(label: str, *, version: str | None = None) -> str:
    """Return the exact source-native HPA IHC label or reject an unavailable label.

    Resolving the label independently of a gene query distinguishes a recognized
    tissue with no observation for that gene from a label absent from the source.
    """
    concrete_version = reference_data.resolve_version("hpa_normal_tissue", version)
    native_labels = hpa_normal_tissue_labels(concrete_version)
    normalized = str(label).strip().casefold()
    matches = [
        native_label for native_label in native_labels if native_label.casefold() == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    raise SafetyTissueResolutionError(
        f"{label!r} is not an HPA IHC tissue label in hpa_normal_tissue@{concrete_version}"
    )


def _strip_version(gene_id: str) -> str:
    return str(gene_id).split(".")[0]


def gene_tissue_ntpm(gene_id: str) -> dict[str, float]:
    """``{tissue (lowercased): nTPM}`` of normal RNA expression for one gene
    (unversioned Ensembl ID)."""
    gid = _strip_version(gene_id)
    df = hpa_rna_consensus()
    sub = df[df["Gene"].astype(str).map(_strip_version) == gid]
    return {str(t).strip().lower(): float(v) for t, v in zip(sub["Tissue"], sub["nTPM"])}


def gene_cell_type_ntpm(gene_id: str) -> dict[str, float]:
    """``{cell_type (lowercased): nTPM}`` of single-cell RNA for one gene."""
    gid = _strip_version(gene_id)
    df = hpa_single_cell()
    sub = df[df["Gene"].astype(str).map(_strip_version) == gid]
    return {str(t).strip().lower(): float(v) for t, v in zip(sub["Cell type"], sub["nTPM"])}


def gene_protein_tissues(gene_id: str, *, levels=("Low", "Medium", "High")) -> set[str]:
    """Tissues (lowercased) where the gene has detected IHC protein at one of
    ``levels`` (default Low/Medium/High)."""
    gid = _strip_version(gene_id)
    df = hpa_normal_tissue()
    sub = df[df["Gene"].astype(str).map(_strip_version) == gid]
    sub = sub[sub["Level"].astype(str).isin(levels)]
    return {str(t).strip().lower() for t in sub["Tissue"]}
