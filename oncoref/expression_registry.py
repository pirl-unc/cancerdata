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

"""Registry of every cohort-level expression data source.

The bundled ``expression_sources.yaml`` is the single source of truth for which
upstream datasets feed each cancer cohort — Treehouse, BEAT-AML, GDC/TCGA,
CLL-map, MMRF CoMMpass, TARGET, GEO series, recount3, … — with the provenance a
re-runner needs (project/accession/url, native unit, library prep, builder,
citation, expected size). It drives the legible ``oncoref expression-sources`` listing
and (incrementally) the fetch + regeneration of the per-cohort summaries.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"
_REGISTRY_PATH = _DATA_DIR / "expression_sources.yaml"


@dataclass(frozen=True)
class ExpressionSource:
    """One upstream expression dataset feeding one or more cancer cohorts."""

    id: str
    category: str
    cancer_codes: tuple[str, ...]
    source_type: str
    builder: str | None = None
    builder_args: tuple[str, ...] = ()
    external_build_exemption: str | None = None
    routed_samples_may_overlap: bool = False
    project_id: str | None = None
    project_ids: tuple[str, ...] = ()
    accession: str | None = None
    url: str | None = None
    unit: str | None = None
    expected_size_gb: float | None = None
    citation: str | None = None
    special_handling: str | None = None
    recount3_srp: str | None = None
    source_cohort: str | None = None
    library_prep: str | None = None
    source_project: str | None = None
    source_version: str | None = None
    tumor_origin: str | None = None
    metastasis_site: str | None = None
    processing_pipeline: str | None = None
    notes: str | None = None
    access_level: str | None = None
    acquisition_status: str | None = None
    data_access_url: str | None = None
    molecular_annotation_status: str | None = None
    source_pmid: str | None = None


def _coerce_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value, *, field: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be a YAML boolean")


def _clean(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def expression_source_registry_path() -> Path:
    """Filesystem path to the bundled ``expression_sources.yaml`` registry.

    This is the stable public path helper for downstream build tooling that
    needs to pass oncoref's source registry to a subprocess or watcher. Prefer
    :func:`expression_source_registry_entries` for ordinary Python access.
    """
    return _REGISTRY_PATH


def expression_source_registry_text() -> str:
    """Raw text of the bundled ``expression_sources.yaml`` registry."""
    return _REGISTRY_PATH.read_text()


def expression_source_registry_entries(
    source_type: str | Iterable[str] | None = None,
) -> tuple[dict, ...]:
    """Raw source-registry entries from ``expression_sources.yaml``.

    The typed :func:`expression_sources` view intentionally exposes the stable
    inspection fields. This helper preserves the full YAML dictionaries for
    build tools that need source-specific fields such as ``file_name``,
    ``sample_filter``, or ``sample_to_cancer_code``. ``source_type`` optionally
    filters to one or more source families, e.g. ``"geo-matrix"``.
    """
    import yaml

    doc = yaml.safe_load(expression_source_registry_text()) or {}
    entries = tuple(dict(entry) for entry in doc.get("sources", []))
    if source_type is None:
        return entries
    wanted = {source_type} if isinstance(source_type, str) else set(source_type)
    return tuple(entry for entry in entries if entry.get("source_type") in wanted)


@lru_cache(maxsize=1)
def load_registry() -> tuple[ExpressionSource, ...]:
    """Parse the bundled ``expression_sources.yaml`` into ``ExpressionSource``s."""
    out = []
    for entry in expression_source_registry_entries():
        out.append(
            ExpressionSource(
                id=str(entry["id"]),
                category=str(entry.get("category", "expression")),
                cancer_codes=_coerce_tuple(entry.get("cancer_codes")),
                source_type=str(entry.get("source_type", "")),
                builder=_clean(entry.get("builder")),
                builder_args=_coerce_tuple(entry.get("builder_args")),
                external_build_exemption=_clean(entry.get("external_build_exemption")),
                routed_samples_may_overlap=_coerce_bool(
                    entry.get("routed_samples_may_overlap"),
                    field=f"{entry['id']}.routed_samples_may_overlap",
                ),
                project_id=_clean(entry.get("project_id")),
                project_ids=_coerce_tuple(entry.get("project_ids")),
                accession=_clean(entry.get("accession")),
                url=_clean(entry.get("url")),
                unit=_clean(entry.get("unit")),
                expected_size_gb=_coerce_float(entry.get("expected_size_gb")),
                citation=_clean(entry.get("citation")),
                source_pmid=_clean(entry.get("source_pmid")),
                special_handling=_clean(entry.get("special_handling")),
                recount3_srp=_clean(entry.get("recount3_srp")),
                source_cohort=_clean(entry.get("source_cohort")),
                library_prep=_clean(entry.get("library_prep")),
                source_project=_clean(entry.get("source_project")),
                source_version=_clean(entry.get("source_version")),
                tumor_origin=_clean(entry.get("tumor_origin")),
                metastasis_site=_clean(entry.get("metastasis_site")),
                processing_pipeline=_clean(entry.get("processing_pipeline")),
                notes=_clean(entry.get("notes")),
                access_level=_clean(entry.get("access_level")),
                acquisition_status=_clean(entry.get("acquisition_status")),
                data_access_url=_clean(entry.get("data_access_url")),
                molecular_annotation_status=_clean(entry.get("molecular_annotation_status")),
            )
        )
    return tuple(out)


def expression_sources() -> tuple[ExpressionSource, ...]:
    """Every registered expression source."""
    return load_registry()


def expression_source(source_id: str) -> ExpressionSource | None:
    """Look up one source by id."""
    for s in load_registry():
        if s.id == source_id:
            return s
    return None


def sources_for_cancer_code(code: str) -> list[ExpressionSource]:
    """Every source that feeds a given cancer code."""
    return [s for s in load_registry() if code in s.cancer_codes]


def expression_sources_df() -> pd.DataFrame:
    """The registry as a tabular view (one row per source) for inspection."""
    return pd.DataFrame(
        {
            "id": s.id,
            "source_type": s.source_type,
            "cancer_codes": ";".join(s.cancer_codes),
            "n_codes": len(s.cancer_codes),
            "source_cohort": s.source_cohort,
            "source_project": s.source_project,
            "source_version": s.source_version,
            "unit": s.unit,
            "library_prep": s.library_prep,
            "project_id": ";".join(s.project_ids) or s.project_id or s.accession or s.recount3_srp,
            "tumor_origin": s.tumor_origin,
            "metastasis_site": s.metastasis_site,
            "processing_pipeline": s.processing_pipeline,
            "notes": s.notes,
            "access_level": s.access_level,
            "acquisition_status": s.acquisition_status,
            "data_access_url": s.data_access_url,
            "molecular_annotation_status": s.molecular_annotation_status,
            "expected_size_gb": s.expected_size_gb,
            "citation": s.citation,
            "source_pmid": s.source_pmid,
            "builder": s.builder,
            "builder_args": " ".join(s.builder_args),
            "external_build_exemption": s.external_build_exemption,
            "routed_samples_may_overlap": s.routed_samples_may_overlap,
        }
        for s in load_registry()
    )


def expression_source_candidates(cancer_code: str | None = None) -> pd.DataFrame:
    """Candidate expression sources per cancer type (``cancer_code``,
    ``source_status``, ``reference_code``, ``source_project``, ``accession``,
    ``assay``, ``processing_plan``, …) — the build-planning view of which cohort
    could back a type's reference. Filtered to one code when given. Defensive copy."""
    from .load_dataset import get_data

    df = get_data("cancer-expression-source-candidates").copy()
    if cancer_code is not None:
        from .cancer_types import resolve_cancer_type

        df = df[df["cancer_code"].astype(str) == resolve_cancer_type(cancer_code)].reset_index(
            drop=True
        )
    return df
