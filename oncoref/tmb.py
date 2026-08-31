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

"""Tumor mutational burden (TMB) reference data, keyed by cancer-type code."""

from __future__ import annotations

import pandas as pd

from .cancer_types import cancer_evidence_source_code, cancer_type_registry, resolve_cancer_type
from .load_dataset import get_data

_TMB_EVIDENCE_OVERRIDES = {
    # MSI-H/dMMR colorectal estimates are published at the CRC source scope; anatomical
    # children such as COAD_MSI/READ_MSI resolve through this row rather than duplicating
    # the same source estimate.
    "CRC_MSI": {
        "estimate_type": "published_median",
        "source_scope": "aggregate_source",
    },
    # The cited GEP-NEN source is pooled across primary sites and WHO grades.
    # Preserve the source audit as explicit missing site-specific estimates.
    "NET_MIDGUT": {
        "estimate_type": "unknown",
        "source_scope": "source_rejected_for_site_specific_value",
        "missing_reason": "no_supported_site_specific_median",
    },
    # These broad entities have subtype/site evidence but no defensible median
    # over the complete aggregate represented by the ontology code.
    "RCC": {
        "estimate_type": "unknown",
        "source_scope": "subtype_sources_not_aggregated",
        "missing_reason": "no_supported_aggregate_median",
    },
    "THYM_EPITHELIAL": {
        "estimate_type": "unknown",
        "source_scope": "subtype_sources_not_aggregated",
        "missing_reason": "source_reports_subtype_medians_only",
    },
    "NEN": {
        "estimate_type": "unknown",
        "source_scope": "source_rejected_for_metric_mismatch",
        "missing_reason": "source_reports_mean_not_median",
    },
    "NET": {
        "estimate_type": "unknown",
        "source_scope": "source_rejected_for_metric_mismatch",
        "missing_reason": "source_reports_mean_not_median",
    },
    "NEC": {
        "estimate_type": "unknown",
        "source_scope": "source_rejected_for_metric_mismatch",
        "missing_reason": "source_reports_mean_not_median",
    },
    "NEC_LUNG": {
        "estimate_type": "unknown",
        "source_scope": "subtype_sources_not_aggregated",
        "missing_reason": "no_supported_aggregate_median",
    },
    "MPN": {
        "estimate_type": "unknown",
        "source_scope": "no_direct_source",
        "missing_reason": "no_published_per_mb_median_curated",
    },
    "CML": {
        "estimate_type": "unknown",
        "source_scope": "no_direct_source",
        "missing_reason": "no_published_per_mb_median_curated",
    },
    "KIRP": {"estimate_type": "approximate_literature"},
    "UCS": {"estimate_type": "approximate_literature"},
    "SARC_RMS_ERMS": {"estimate_type": "approximate_literature"},
    "SARC_RMS_ARMS": {"estimate_type": "approximate_literature"},
    "WILMS": {"estimate_type": "approximate_literature"},
    "MCL": {"estimate_type": "approximate_literature"},
    "HL": {"estimate_type": "approximate_literature"},
    "BL": {"estimate_type": "approximate_literature"},
    "T_ALL": {"estimate_type": "approximate_literature"},
    "MTC": {"estimate_type": "panel_inferred"},
    "CRANIO": {"estimate_type": "small_n"},
    "HCL": {"estimate_type": "small_n"},
    "UCEC_POLE": {"estimate_type": "order_of_magnitude"},
    # TCGA-SARC is a soft-tissue sarcoma cohort. It does not span the full oncoref
    # SARC member-union scope, which also includes bone sarcomas and RMS.
    "SARC": {"source_scope": "soft_tissue_sarcoma_subset"},
    "SARC_CIC": {"source_scope": "renal_cic_rearranged_sarcoma_proxy"},
}


def cancer_tmb_df():
    """Return the curated ``cancer-tmb.csv`` reference: median tumor mutational
    burden (mut/Mb) per cancer-type code, with a per-row published source/PMID
    and a confidence flag.

    Cohorts with no defensible published per-Mb median are present with a blank
    ``median_tmb_mut_mb`` (and a ``confidence`` of ``none``) so the gap is
    explicit rather than silently absent. Values mix WES-anchored medians
    (Lawrence 2013) with panel-based medians (Chalmers 2017) and disease-specific
    studies; see the ``source``/``notes`` columns — panel and WES TMB are not
    strictly comparable in the low-TMB range."""
    df = get_data("cancer-tmb").copy()
    evidence = [_tmb_evidence_fields(row) for _, row in df.iterrows()]
    for col in ("estimate_type", "source_scope", "missing_reason"):
        df[col] = [record[col] for record in evidence]
    return df


def _tmb_evidence_fields(row) -> dict[str, object]:
    code = str(row.get("cancer_code", ""))
    override = _TMB_EVIDENCE_OVERRIDES.get(code, {})
    value = row.get("median_tmb_mut_mb")
    if pd.isna(value):
        return {
            "estimate_type": override.get("estimate_type", "unknown"),
            "source_scope": override.get("source_scope", "no_direct_source"),
            "missing_reason": override.get("missing_reason", "no_published_per_mb_median_curated"),
        }
    return {
        "estimate_type": override.get("estimate_type", "published_median"),
        "source_scope": override.get("source_scope", "cancer_code_direct"),
        "missing_reason": override.get("missing_reason", float("nan")),
    }


def _tmb_value_map(df=None) -> dict[str, float]:
    df = cancer_tmb_df() if df is None else df
    vals = df.dropna(subset=["median_tmb_mut_mb"])
    return dict(zip(vals["cancer_code"].astype(str), vals["median_tmb_mut_mb"].astype(float)))


def _parent_code(code: str, registry) -> str | None:
    if code not in registry.index:
        return None
    parent = registry.loc[code].get("parent_code", "")
    if pd.isna(parent):
        return None
    parent = str(parent).strip()
    return parent or None


def _public_value(value):
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _record_from_row(row, *, requested_code: str, resolved_code: str, inheritance_kind: str):
    record = {key: _public_value(row[key]) for key in row.index}
    record["requested_cancer_code"] = requested_code
    record["resolved_cancer_code"] = resolved_code
    record["inheritance_kind"] = inheritance_kind
    record["is_inherited_evidence"] = requested_code != resolved_code
    record["has_tmb_source"] = True
    return record


def _resolve_tmb_row(requested_code: str, *, inherit: bool):
    df = cancer_tmb_df()
    values = _tmb_value_map(df)
    rows = df.set_index("cancer_code", drop=False)

    if requested_code in values:
        return requested_code, "direct", rows.loc[requested_code]
    if requested_code in rows.index:
        return requested_code, "direct_missing", rows.loc[requested_code]
    if not inherit:
        return requested_code, "missing", None

    source_code = cancer_evidence_source_code(requested_code)
    if source_code != requested_code and source_code in values:
        return source_code, "source_scope", rows.loc[source_code]

    registry = cancer_type_registry().set_index("code")
    cur = _parent_code(requested_code, registry)
    seen = {requested_code}
    while cur and cur not in seen:
        seen.add(cur)
        if cur in values:
            return cur, "ancestor", rows.loc[cur]
        cur = _parent_code(cur, registry)
    return requested_code, "missing", None


def resolve_tmb_source(cancer_type, *, inherit=True) -> dict:
    """Resolve the source row used for a TMB lookup.

    Returns metadata without requiring callers to inspect the raw table:

    - ``requested_cancer_code``: canonical code requested by the caller.
    - ``resolved_cancer_code``: direct/source-scope/ancestor row used, if any.
    - ``inheritance_kind``: ``"direct"``, ``"direct_missing"``, ``"source_scope"``,
      ``"ancestor"``, or ``"missing"``.
    - source/provenance fields from the selected row when available.

    This makes aggregate evidence explicit. For example, ``COAD_MSI`` and
    ``READ_MSI`` resolve through the curated ``CRC_MSI`` TMB row, preserving that the
    source estimate is CRC-level MSI-H/dMMR evidence rather than a colon- or
    rectum-specific median.
    """
    requested_code = resolve_cancer_type(cancer_type)
    resolved_code, inheritance_kind, row = _resolve_tmb_row(requested_code, inherit=inherit)
    if row is None:
        return {
            "requested_cancer_code": requested_code,
            "resolved_cancer_code": None,
            "inheritance_kind": inheritance_kind,
            "is_inherited_evidence": False,
            "has_tmb_source": False,
        }
    return _record_from_row(
        row,
        requested_code=requested_code,
        resolved_code=resolved_code,
        inheritance_kind=inheritance_kind,
    )


def cancer_tmb(cancer_type=None, *, inherit=True):
    """Median TMB (mut/Mb) for one cancer type, or the whole
    ``{code: median_tmb}`` map (codes with no published value omitted).

    ``cancer_type`` is resolved through :func:`resolve_cancer_type`, so aliases
    and display names work. When ``inherit`` (default), a code with no curated
    value of its own inherits its nearest ancestor's TMB by walking the registry
    ``parent_code`` chain — so molecular / histology subtypes (``LUAD_EGFR`` ->
    ``LUAD``, ``SCLC_ASCL1`` -> ``SCLC``, rare ``SARC_*`` -> ``SARC``) resolve
    without a curated row each. Returns ``None`` if neither the code nor any
    ancestor has a value."""
    df = cancer_tmb_df()
    mapping = _tmb_value_map(df)
    if cancer_type is None:
        return mapping
    code = resolve_cancer_type(cancer_type)
    resolved_code, _, row = _resolve_tmb_row(code, inherit=inherit)
    return mapping.get(resolved_code) if row is not None else None


def cancer_tmb_record(cancer_type=None, *, inherit=True):
    """Metadata-bearing TMB lookup.

    Mirrors :func:`cancer_tmb`, but returns the resolved source row as a dict instead
    of only the numeric median. The record includes the derived evidence columns from
    :func:`cancer_tmb_df` plus lookup metadata:

    - ``requested_cancer_code``
    - ``resolved_cancer_code``
    - ``inheritance_kind``
    - ``is_inherited_evidence``

    With ``cancer_type=None`` the returned bulk map contains direct source rows only.
    Use :func:`resolve_tmb_source` for explicit requested-code resolution metadata.
    """
    if cancer_type is None:
        return {
            code: record
            for code in sorted(cancer_tmb())
            if (record := cancer_tmb_record(code, inherit=False))
        }
    record = resolve_tmb_source(cancer_type, inherit=inherit)
    return record if record.get("has_tmb_source") else None


def cancer_frameshift_burden_df():
    """Return the curated ``cancer-frameshift-burden.csv`` reference: per-type
    frameshift-indel burden (``cancer_code``, ``indel_class``, ``indel_score``,
    ``basis``, ``pmid_doi``, ``confidence``, ``notes``). A complement to TMB —
    frameshift indels yield disproportionately many high-affinity neoantigens."""
    return get_data("cancer-frameshift-burden")


def cancer_frameshift_burden(cancer_type=None, *, inherit=True):
    """Frameshift-indel ``indel_score`` for a cancer type (alias-resolved), or the
    full ``{code: score}`` map when ``cancer_type`` is None. Subtypes inherit a
    parent's score if unmapped (mirrors :func:`cancer_tmb`)."""
    df = cancer_frameshift_burden_df()
    scores = {
        str(c): int(s)
        for c, s in zip(df["cancer_code"], df["indel_score"])
        if str(s).strip() and str(s).strip().lower() != "nan"
    }
    if cancer_type is None:
        return scores
    code = resolve_cancer_type(cancer_type)
    if code in scores:
        return scores[code]
    if inherit:
        parent_of = cancer_type_registry().set_index("code")["parent_code"].to_dict()
        cur = code
        seen = set()
        while cur in parent_of and isinstance(parent_of[cur], str) and parent_of[cur]:
            cur = parent_of[cur]
            if cur in seen:
                break
            seen.add(cur)
            if cur in scores:
                return scores[cur]
    return None
