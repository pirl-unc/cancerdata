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

"""Anti-PD-1 monotherapy response (objective response rate) by cancer type."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ._evidence_resolution import evidence_record, resolve_evidence
from .cancer_types import cancer_evidence_source_code, cancer_type_registry, resolve_cancer_type
from .ici import response_anchor_evidence_df
from .load_dataset import _register_derived_cache, get_data


def cancer_apd1_response_df():
    """Return the curated ``cancer-apd1-response.csv`` reference: representative
    objective response rate (ORR, %) to anti-PD-1 **monotherapy**
    (pembrolizumab / nivolumab) per cancer-type code, with the drug, pivotal
    trial, treatment setting, a published source PMID/DOI, and a confidence flag.

    Intended as a per-cancer-type plotting axis (e.g. TMB vs aPD1 ORR, CTA burden
    vs aPD1 ORR). Values are representative anchors, not exact reproducible
    constants — they shift with data cutoff, line of therapy, and biomarker
    selection (PD-L1 / MSI / MMR); the ``setting`` and ``notes`` columns record
    that context. Evidence/provenance fields are joined from the audited ICI estimates
    table, keyed by ``drug_target`` (``PD-1`` / ``PD-L1`` / ``PD-1+CTLA-4``), so
    non-monotherapy fallback anchors remain explicit."""
    return _apd1_response_evidence_frame().copy()


@lru_cache(maxsize=1)
def _apd1_response_evidence_frame():
    """The joined aPD-1 anchor+evidence frame, built once.

    Same shared join as :func:`oncoref.ici.cancer_ici_response_df`, and the same
    rebuild-per-lookup cost: ``_resolve_apd1_response_row`` runs one lookup per
    requested code, and the ``include_inherited`` bulk maps run one per registry code.
    Public callers still receive a defensive copy."""
    return response_anchor_evidence_df(
        get_data("cancer-apd1-response"),
        value_col="apd1_orr_pct",
        regimen_col="drug_target",
    )


_register_derived_cache(_apd1_response_evidence_frame.cache_clear)


@lru_cache(maxsize=1)
def _apd1_valued_rows():
    """The anchored aPD-1 rows only. Cached; treat as read-only."""
    return _apd1_response_evidence_frame().dropna(subset=["apd1_orr_pct"])


_register_derived_cache(_apd1_valued_rows.cache_clear)


def cancer_apd1_response(cancer_type=None, *, inherit=True, include_inherited=False):
    """Anti-PD-1 monotherapy ORR (%) for one cancer type, or the whole
    ``{code: orr_pct}`` map. ``cancer_type`` is resolved through
    :func:`resolve_cancer_type`; with ``inherit`` (default) a code with no
    curated row of its own inherits its nearest ancestor's value via the registry
    ``parent_code`` chain. Returns ``None`` if neither the code nor any ancestor
    has a value. Mirrors :func:`oncoref.cancer_tmb`.

    With ``cancer_type=None`` the default map contains direct source rows only. Pass
    ``include_inherited=True`` to expand across registry codes with the same resolver
    used for individual lookups, so source-scoped children such as ``COAD_MSI`` and
    ``READ_MSI`` are included with inherited values.
    """
    vals = _apd1_valued_rows()
    mapping = dict(zip(vals["cancer_code"].astype(str), vals["apd1_orr_pct"].astype(float)))
    if cancer_type is None:
        if include_inherited:
            out = {}
            codes = sorted(set(cancer_type_registry()["code"].astype(str)) | set(mapping))
            for code in codes:
                value = cancer_apd1_response(code, inherit=inherit)
                if value is not None:
                    out[code] = value
            return out
        return mapping
    code = resolve_cancer_type(cancer_type)
    _, _, row = _resolve_apd1_response_row(code, inherit=inherit)
    return None if row is None else float(row["apd1_orr_pct"])


def _record_from_row(row, *, requested_code: str, resolved_code: str, inheritance_kind: str):
    record = evidence_record(
        row,
        requested_code=requested_code,
        resolved_code=resolved_code,
        inheritance_kind=inheritance_kind,
    )
    record["selected_regimen"] = record.get("drug_target")
    record["selected_drug_target"] = record.get("drug_target")
    return record


def _matching_row(df: pd.DataFrame, code: str):
    hit = df[df["cancer_code"].astype(str) == code]
    return None if hit.empty else hit.iloc[0]


def _resolve_apd1_response_row(requested_code: str, *, inherit: bool):
    df = _apd1_valued_rows()
    resolution = resolve_evidence(
        requested_code,
        direct_lookup=lambda code: _matching_row(df, code),
        source_code_for=cancer_evidence_source_code,
        parent_by_code=cancer_type_registry().set_index("code")["parent_code"],
        inherit=inherit,
    )
    return resolution.resolved_code, resolution.inheritance_kind, resolution.payload


def resolve_apd1_response_source(cancer_type, *, inherit=True) -> dict:
    """Resolve the evidence source row used for an anti-PD-1 response lookup.

    Returns lookup metadata without reducing the result to a numeric ORR:
    ``requested_cancer_code``, ``resolved_cancer_code``, ``inheritance_kind`` and
    source/provenance fields from :func:`cancer_apd1_response_df` when available.
    Source-scoped molecular children such as ``COAD_MSI`` and ``READ_MSI`` resolve
    through the curated ``CRC_MSI`` row while preserving that the evidence is inherited
    from an aggregate/source-scope estimate.
    """
    requested_code = resolve_cancer_type(cancer_type)
    resolved_code, inheritance_kind, row = _resolve_apd1_response_row(
        requested_code, inherit=inherit
    )
    if row is None:
        return {
            "requested_cancer_code": requested_code,
            "resolved_cancer_code": None,
            "inheritance_kind": inheritance_kind,
            "is_inherited_evidence": False,
            "selected_regimen": None,
            "selected_drug_target": None,
            "has_apd1_response_source": False,
        }
    record = _record_from_row(
        row,
        requested_code=requested_code,
        resolved_code=resolved_code,
        inheritance_kind=inheritance_kind,
    )
    record["has_apd1_response_source"] = True
    return record


def cancer_apd1_response_record(cancer_type=None, *, inherit=True, include_inherited=False):
    """Metadata-bearing anti-PD-1 objective response lookup.

    Mirrors :func:`cancer_apd1_response`, but returns the resolved anchor row as a
    dict instead of only the ORR value. The record includes joined evidence fields
    from the audited ICI estimates table plus requested/resolved-code metadata. With
    ``cancer_type=None`` the returned map contains direct source rows only by default;
    pass ``include_inherited=True`` to expand across registry codes with inherited
    source metadata.
    """
    if cancer_type is None:
        df = _apd1_valued_rows()
        direct_codes = set(df["cancer_code"].astype(str).unique())
        codes = (
            sorted(set(cancer_type_registry()["code"].astype(str)) | direct_codes)
            if include_inherited
            else sorted(direct_codes)
        )
        record_inherit = inherit if include_inherited else False
        return {
            str(code): record
            for code in codes
            if (record := cancer_apd1_response_record(code, inherit=record_inherit))
        }

    requested_code = resolve_cancer_type(cancer_type)
    resolved_code, inheritance_kind, row = _resolve_apd1_response_row(
        requested_code, inherit=inherit
    )
    if row is None:
        return None
    return _record_from_row(
        row,
        requested_code=requested_code,
        resolved_code=resolved_code,
        inheritance_kind=inheritance_kind,
    )
