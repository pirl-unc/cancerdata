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

"""Immune-checkpoint-inhibitor (ICI) response (ORR) by cancer type and regimen.

Generalizes the anti-PD-1 layer (:mod:`oncoref.apd1`) to **all three checkpoint
regimens, each kept as a distinct source of response data**:

- ``"PD-1"``        — anti-PD-1 monotherapy (pembrolizumab / nivolumab / cemiplimab)
- ``"PD-L1"``       — anti-PD-L1 monotherapy (atezolizumab / durvalumab / avelumab)
- ``"PD-1+CTLA-4"`` — anti-PD-1 + anti-CTLA-4 combination (nivolumab + ipilimumab)

Unlike the representative one-row-per-cancer ``cancer-apd1-response.csv``, the ICI
table (``cancer-ici-response.csv``) is a **long table**: a cancer type can carry a
value for more than one regimen (e.g. melanoma under both anti-PD-1 mono and the
ipi+nivo doublet), so regimens can be compared within a cancer. Every value is a
representative ORR anchor from a pivotal trial, with a citation — not an exact
reproducible constant (it shifts with data cutoff, line of therapy, and PD-L1 / MSI
selection; the ``setting`` and ``notes`` columns record that context).

:func:`cancer_ici_response` exposes the **fallback resolution** the analysis layer
usually wants — prefer anti-PD-1 monotherapy, fall back to anti-PD-L1 where that is
missing, then to the combination — via the default :data:`REGIMEN_FALLBACK` order;
pass ``regimen=`` to pin a single regimen instead, or ``fallback=False`` to get the
full per-regimen mapping.

Curation & pooling criteria
---------------------------
The wider evidence base (``cancer-ici-response-estimates.csv``, exposed by
:func:`cancer_ici_response_estimates_df` and pooled by :func:`pooled_ici_response`)
follows three rules that matter when curating new trials or interpreting a pooled value:

1. **Reported, computed, and derived values stay distinct.** ``value_basis`` records
   whether a value was reported directly, computed from reported counts, inferred
   from outcomes, combined across cohorts, retained as overlapping context, or
   curator-modeled. :func:`pooled_ici_response` accepts direct reported values and
   same-cohort count calculations; it excludes inferred, cross-cohort, contextual,
   and modeled values.

   The clearest modeled case is the "all-comer" ORR for MSI/MMR-dependent cancers:
   ``READ`` 5%, ``COAD`` 5%, and ``UCEC`` 8% are prevalence-weighted blends of the
   MSI-H/dMMR responders and MSS/pMMR non-responders because the pivotal trials enroll
   only biomarker-selected subtypes. These rows carry
   ``value_basis="derived_blend"`` and ``source_verified=False`` because no paper
   reports the modeled result. The blend is reconstructable from its components:
   ``all_comer ≈ ORR_MSI · p_dMMR + ORR_MSS · (1 − p_dMMR)`` (COAD: 43.8·0.13 ≈ 5.7%;
   READ: 43.8·0.07 ≈ 3.1%; UCEC: 48·0.20 + 7·0.80 ≈ 15%, using the KEYNOTE-158 dMMR/pMMR
   cohorts at an advanced-EC dMMR prevalence ~20%). When adding such an anchor,
   keep the reported subtype values in the ``<code>_MSI`` / ``<code>_MSS`` rows and record
   the prevalence weighting in ``notes`` — never cite a paper that does not contain the
   blended number.

2. **Never double-count patients.** A single trial routinely reports an all-comer cohort
   AND its own biomarker subgroup (e.g. ``BLCA`` KEYNOTE-052 all-comers + the CPS≥10 subset;
   ``LUAD`` all-comers + PD-L1≥50%). Those share patients, so summing their denominators
   inflates ``n``. Each estimate row therefore carries a ``role``: ``"primary"`` (the one
   representative cited setting) or ``"alternate"`` (other trials/subgroups). Pool only
   rows describing the **same population and line of therapy**, and never an all-comer
   cohort together with a subgroup drawn from it. ``pooled_ici_response`` does *not*
   auto-dedupe overlapping subgroups — it returns the full ``sources`` list and
   ``value_range`` so the overlap stays visible; ``include_alternates=False`` restricts
   the pool to ``primary`` rows (one per cancer+regimen), which can never overlap.

3. **Comparability.** ORR shifts with line of therapy, PD-L1/MSI selection, and data
   cutoff; medians (PFS/OS/DOR) cannot be pooled at all without patient-level data. Treat
   ``value_range`` as a heterogeneity check before trusting a single pooled number.
"""

from __future__ import annotations

import math
from functools import lru_cache

import pandas as pd

from .cancer_types import cancer_evidence_source_code, cancer_type_registry, resolve_cancer_type
from .load_dataset import _register_derived_cache, get_data

#: Response-proportion endpoints that can be responder-weighted-pooled (each needs a
#: responder count and a denominator n). Time-to-event medians and landmark rates are
#: deliberately excluded — see :func:`pooled_ici_response`.
PROPORTION_METRICS: tuple[str, ...] = ("ORR", "CRR", "DCR", "PR")

#: Values that are useful audit context but should not enter pooled estimates.
NON_POOLABLE_VALUE_BASIS: frozenset[str] = frozenset(
    {
        "derived_blend",
        "derived_cross_cohort",
        "inferred_from_outcomes",
        "reported_context",
    }
)

#: Regimen tags in preference order — the default fallback when no regimen is pinned:
#: anti-PD-1 monotherapy first, then anti-PD-L1, then the anti-PD-1+anti-CTLA-4 doublet.
REGIMEN_FALLBACK: tuple[str, ...] = ("PD-1", "PD-L1", "PD-1+CTLA-4")

#: Human-readable label for each regimen tag.
REGIMEN_LABELS = {
    "PD-1": "anti-PD-1 monotherapy",
    "PD-L1": "anti-PD-L1 monotherapy",
    "PD-1+CTLA-4": "anti-PD-1 + anti-CTLA-4",
    "PD-1+HMA": "anti-PD-1 + hypomethylating agent",
}

REGIMEN_CLASSES = {
    "PD-1": "anti_pd1_monotherapy",
    "PD-L1": "anti_pdl1_monotherapy",
    "PD-1+CTLA-4": "anti_pd1_ctla4_combination",
    "PD-1+HMA": "anti_pd1_hma_combination",
}


def _mixture_cohort_code_set() -> frozenset[str]:
    """Registry codes whose evidence rows represent aggregate/source-scope cohorts."""
    reg = cancer_type_registry()
    if "mixture_cohort" not in reg.columns:
        return frozenset()
    flag = reg["mixture_cohort"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    return frozenset(reg.loc[flag, "code"].astype(str))


#: Reviewed provenance for curated rows that carry no representative ORR. These are
#: audited gaps, not unreviewed holes: a reviewer has recorded *why* no single ORR
#: describes the code. Mirrors ``oncoref.tmb._TMB_EVIDENCE_OVERRIDES``.
_ICI_EVIDENCE_OVERRIDES = {
    # Checkpoint response in colorectal cancer is determined by mismatch-repair
    # status, not by the anatomical aggregate: MSI-H/dMMR disease responds and MSS
    # disease essentially does not. A pooled CRC ORR would average two populations
    # that must never be averaged, so the aggregate stays an explicit gap and callers
    # are directed to the stratified rows.
    "CRC": {
        "source_scope": "subtype_sources_not_aggregated",
        "missing_reason": "response_is_mmr_stratified_not_aggregate",
    },
    # Clear-cell and non-clear-cell renal carcinoma have separate curated anchors
    # (KIRC CheckMate 025 / 214, RCC_NCC KEYNOTE-427 cohort B) with materially
    # different response; the histology-spanning aggregate has no single trial.
    "RCC": {
        "source_scope": "subtype_sources_not_aggregated",
        "missing_reason": "no_supported_aggregate_orr",
    },
    # Breast checkpoint evidence is receptor-subtype evidence: the curated anchors sit
    # on BRCA_Basal (KEYNOTE-086, PCD4989g). Hormone-receptor-positive disease has no
    # comparable single-agent anchor, so no all-comer breast ORR is defensible.
    "BRCA": {
        "source_scope": "subtype_sources_not_aggregated",
        "missing_reason": "response_is_receptor_subtype_stratified",
    },
    # SARC028 and the AcSe/DART programmes report per-histology response that ranges
    # from 0% (LMS, EWS, GIST) to 23-25% (UPS, SMARCA4); the sarcoma umbrella is a
    # member union over those histologies and has no meaningful pooled ORR.
    "SARC": {
        "source_scope": "subtype_sources_not_aggregated",
        "missing_reason": "response_is_histology_stratified",
    },
    # The STAD anchor is an all-comer gastric ORR, and MSI-H/dMMR gastric disease
    # responds substantially better, so inheriting it understates the subtype. Curated
    # alongside the STAD_MSI TMB gap so the code is not half-audited: unknown TMB but
    # a confidently inherited all-comer ORR.
    "STAD_MSI": {
        "source_scope": "source_rejected_for_subtype_value",
        "missing_reason": "no_supported_subtype_orr",
    },
}


def _evidence_type(value_basis) -> str:
    basis = str(value_basis).strip()
    return {
        "computed_from_counts": "computed_from_counts",
        "derived_blend": "derived_blend",
        "derived_cross_cohort": "derived_cross_cohort",
        "inferred_from_outcomes": "inferred_from_outcomes",
        "reported_context": "reported_context",
    }.get(basis, "direct_reported")


def _source_scope(code, value_basis, mixture_codes: frozenset[str]) -> str:
    basis = str(value_basis).strip()
    if basis in {"derived_blend", "derived_cross_cohort"}:
        return basis
    if str(code) in mixture_codes:
        return "aggregate_source"
    return "direct_source"


def response_anchor_evidence_df(
    anchors,
    *,
    value_col: str,
    regimen_col: str = "regimen",
    gap_codes=frozenset(),
):
    """Append evidence/provenance fields to compact representative ICI anchors.

    The compact ICI and aPD1 tables intentionally keep only the plotting anchor fields.
    The richer audit table (``cancer-ici-response-estimates.csv``) has one primary ORR
    row for every compact anchor. This helper joins that source row back onto the
    anchor table so public accessors expose both oncoref's trial identity and
    pirlygenes-style denominator/evidence semantics without duplicating curation.
    """
    df = anchors.copy()
    estimates = cancer_ici_response_estimates_df()
    primary_orr = estimates[
        (estimates["metric"].astype(str).str.upper() == "ORR")
        & (estimates["role"].astype(str) == "primary")
    ].copy()
    primary_orr = primary_orr[
        [
            "cancer_code",
            "regimen",
            "estimate_id",
            "ref",
            "source_locator",
            "source_locator_status",
            "source_endpoint_label",
            "source_population_label",
            "setting",
            "source_n",
            "metric",
            "value",
            "value_status",
            "unit",
            "ci_low",
            "ci_low_status",
            "ci_high",
            "ci_high_status",
            "ci_basis",
            "metric_n",
            "responders",
            "source_verified",
            "value_basis",
        ]
    ].rename(
        columns={
            "ref": "source_anchor",
            "setting": "endpoint_population",
            "source_n": "source_n",
            "metric": "response_metric",
            "value": "_response_value",
            "unit": "response_unit",
            "ci_low": "response_ci_low",
            "ci_low_status": "response_ci_low_status",
            "ci_high": "response_ci_high",
            "ci_high_status": "response_ci_high_status",
            "ci_basis": "response_ci_basis",
            "metric_n": "response_denominator",
            "responders": "response_numerator",
            "estimate_id": "source_estimate_id",
            "value_status": "response_value_status",
        }
    )
    merged = df.merge(
        primary_orr,
        how="left",
        left_on=["cancer_code", regimen_col],
        right_on=["cancer_code", "regimen"],
        suffixes=("", "_evidence"),
        validate="one_to_one",
    )
    if regimen_col != "regimen":
        merged = merged.drop(columns=["regimen"])
    # A curated gap row deliberately carries no anchor value, so it has no primary ORR
    # estimate to join. Being blank is *not* enough to earn that exemption: the code
    # must also be declared in the caller's reviewed gap set, so an accidentally blank
    # value still raises instead of being silently promoted to an "audited" gap.
    is_gap = merged[value_col].isna() & merged["cancer_code"].astype(str).isin(gap_codes)
    missing = merged["response_metric"].isna() & ~is_gap
    if missing.any():
        # fillna before astype: a blank regimen is a float NaN under the string dtype
        # and astype(str) leaves it as one, which would break the join below.
        cells = (
            merged.loc[missing, ["cancer_code", regimen_col]]
            .fillna("")
            .astype(str)
            .agg("/".join, axis=1)
        )
        raise ValueError(f"missing primary ORR evidence rows for: {', '.join(cells)}")
    # A gap is a statement about the cancer code, not about one regimen, so a gap row
    # must not name a regimen — that keeps the code-keyed gap lookup unambiguous and
    # stops a per-regimen blank from silently suppressing the code's other regimens.
    named_regimen = is_gap & merged[regimen_col].notna()
    if named_regimen.any():
        codes = sorted(set(merged.loc[named_regimen, "cancer_code"].astype(str)))
        raise ValueError(f"curated gap rows must leave regimen blank: {', '.join(codes)}")
    valued_gap_codes = set(merged.loc[merged[value_col].notna(), "cancer_code"].astype(str)) & set(
        merged.loc[is_gap, "cancer_code"].astype(str)
    )
    if valued_gap_codes:
        raise ValueError(
            f"codes cannot be both valued and an audited gap: {', '.join(sorted(valued_gap_codes))}"
        )

    mixture_codes = _mixture_cohort_code_set()
    merged["source_anchor"] = merged["source_anchor"].where(merged["source_anchor"].notna(), None)
    merged["response_metric"] = merged["response_metric"].astype(str).str.upper()
    merged["response_value_matches_anchor"] = (
        merged[value_col].astype(float) - merged["_response_value"].astype(float)
    ).abs() <= 2.0
    merged["therapy_regimen_class"] = (
        merged[regimen_col].map(REGIMEN_CLASSES).fillna("other_ici_regimen")
    )
    merged["evidence_type"] = merged["value_basis"].map(_evidence_type)
    merged["histology_match"] = merged["evidence_type"].map(
        {
            "direct_reported": "direct",
            "computed_from_counts": "direct",
            "inferred_from_outcomes": "direct",
            "reported_context": "context",
            "derived_blend": "derived",
            "derived_cross_cohort": "derived",
        }
    )
    merged["is_direct_cancer_code_evidence"] = merged["evidence_type"].isin(
        {"direct_reported", "computed_from_counts", "inferred_from_outcomes"}
    )
    merged["evidence_source_code"] = merged["cancer_code"].astype(str)
    merged["source_scope"] = [
        _source_scope(code, basis, mixture_codes)
        for code, basis in zip(merged["cancer_code"], merged["value_basis"])
    ]
    merged["missing_reason"] = None
    if is_gap.any():
        _apply_gap_evidence(merged, is_gap)
    return merged.drop(columns=["_response_value"])


def _apply_gap_evidence(merged, is_gap) -> None:
    """Rewrite the derived evidence fields for curated rows that carry no ORR.

    A gap row is a reviewed statement that no defensible representative ORR exists for
    the code, so it must not pick up the ``direct_reported`` defaults that the
    ``value_basis`` mapping assigns to an empty cell.
    """
    merged.loc[is_gap, "response_metric"] = None
    merged.loc[is_gap, "evidence_type"] = "unknown"
    merged.loc[is_gap, "histology_match"] = None
    merged.loc[is_gap, "is_direct_cancer_code_evidence"] = False
    merged.loc[is_gap, "response_value_matches_anchor"] = False
    # A gap row names no regimen and cites no evidence, so it must not keep the
    # regimen class the fillna default assigned it, nor claim itself as its own
    # evidence source.
    merged.loc[is_gap, "therapy_regimen_class"] = None
    merged.loc[is_gap, "evidence_source_code"] = None
    for position, code in merged.loc[is_gap, "cancer_code"].items():
        override = _ICI_EVIDENCE_OVERRIDES[str(code)]
        merged.at[position, "source_scope"] = override["source_scope"]
        merged.at[position, "missing_reason"] = override["missing_reason"]


def cancer_ici_response_df():
    """The curated ``cancer-ici-response.csv`` long table: one row per
    (``cancer_code``, ``regimen``) with the representative ORR (%), drug, pivotal trial
    (split into ``trial_name`` / ``trial_alias`` / ``trial_nct`` — the acronym, the
    distinct protocol/sponsor code if any, and the ClinicalTrials.gov id), setting,
    source PMID/DOI, confidence, and evidence/provenance fields joined from the
    audited estimates table. The joined fields include ``source_estimate_id`` (the
    exact primary ORR row in ``cancer_ici_response_estimates_df``), source-locator
    extraction status, and structured CI/value status. A cancer type may appear under
    several regimens."""
    return response_anchor_evidence_df(
        get_data("cancer-ici-response"),
        value_col="orr_pct",
        gap_codes=frozenset(_ICI_EVIDENCE_OVERRIDES),
    )


def ici_regimens() -> tuple[str, ...]:
    """The regimen tags, in fallback-preference order."""
    return REGIMEN_FALLBACK


@lru_cache(maxsize=1)
def _regimen_maps() -> dict[str, dict[str, float]]:
    """``{regimen: {cancer_code: orr_pct}}`` from the curated table. Cached; callers
    must treat the result as read-only (copy before mutating)."""
    df = cancer_ici_response_df().dropna(subset=["orr_pct"])
    out: dict[str, dict[str, float]] = {r: {} for r in REGIMEN_FALLBACK}
    for code, regimen, orr in zip(df["cancer_code"], df["regimen"], df["orr_pct"]):
        out.setdefault(str(regimen), {})[str(code)] = float(orr)
    return out


_register_derived_cache(_regimen_maps.cache_clear)


@lru_cache(maxsize=1)
def _gap_rows() -> dict[str, object]:
    """``{cancer_code: row}`` for curated rows with no representative ORR.

    These are the audited ICI gaps. They are excluded from the value maps (there is no
    value) but they still resolve, so a reviewed "no defensible aggregate" is
    distinguishable from a code nobody has looked at yet."""
    df = cancer_ici_response_df()
    gaps = df[df["orr_pct"].isna()]
    return {str(row["cancer_code"]): row for _, row in gaps.iterrows()}


_register_derived_cache(_gap_rows.cache_clear)


def _gap_record(requested_code: str) -> dict:
    """The public record for an audited gap: the curated row plus lookup metadata."""
    record = _record_from_row(
        _gap_rows()[requested_code],
        requested_code=requested_code,
        resolved_code=requested_code,
        inheritance_kind="direct_missing",
    )
    record.update(
        {
            "selected_regimen": None,
            "available_regimens": (),
            "has_ici_response_source": True,
        }
    )
    return record


def _resolve_with_fallback(code: str, maps: dict[str, dict[str, float]], order):
    for regimen in order:
        if code in maps.get(regimen, {}):
            return maps[regimen][code], regimen
    return None, None


def _parent_code(code: str, registry) -> str | None:
    if code not in registry.index:
        return None
    parent = registry.loc[code].get("parent_code", "")
    if pd.isna(parent):
        return None
    parent = str(parent).strip()
    return parent or None


def _bulk_lookup_codes(*, include_inherited: bool = False) -> list[str]:
    df = cancer_ici_response_df().dropna(subset=["orr_pct"])
    direct_codes = {str(code) for code in df["cancer_code"]}
    if not include_inherited:
        return sorted(direct_codes)
    registry_codes = set(cancer_type_registry()["code"].astype(str))
    return sorted(registry_codes | direct_codes)


def _matching_rows(df, code: str, order) -> dict[str, object]:
    rows: dict[str, object] = {}
    sub = df[df["cancer_code"] == code]
    for regimen in order:
        hit = sub[sub["regimen"] == regimen]
        if not hit.empty:
            rows[str(regimen)] = hit.iloc[0]
    return rows


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
    record["selected_regimen"] = record.get("regimen")
    record["inheritance_kind"] = inheritance_kind
    record["is_inherited_evidence"] = requested_code != resolved_code
    return record


def _resolve_ici_response_source(requested_code: str, order, *, inherit: bool):
    df = cancer_ici_response_df().dropna(subset=["orr_pct"])
    direct = _matching_rows(df, requested_code, order)
    if direct:
        return requested_code, "direct", direct

    # A curated gap row is a reviewed decision about *this* code, so it outranks any
    # inherited evidence — exactly as a blank-value TMB row stops the ancestor walk.
    # It is reported whether or not inheritance is enabled, because the review applies
    # to the requested code itself; ``tmb._resolve_tmb_row`` makes the same ordering.
    if requested_code in _gap_rows():
        return requested_code, "direct_missing", {}
    if not inherit:
        return requested_code, "direct", direct

    source_code = cancer_evidence_source_code(requested_code)
    if source_code != requested_code:
        source = _matching_rows(df, source_code, order)
        if source:
            return source_code, "source_scope", source

    registry = cancer_type_registry().set_index("code")
    cur = _parent_code(requested_code, registry)
    seen = {requested_code}
    while cur and cur not in seen:
        seen.add(cur)
        inherited = _matching_rows(df, cur, order)
        if inherited:
            return cur, "ancestor", inherited
        cur = _parent_code(cur, registry)
    return requested_code, "missing", {}


def _resolve_ici_response_rows(requested_code: str, order, *, inherit: bool):
    return _resolve_ici_response_source(requested_code, order, inherit=inherit)


def resolve_ici_response_source(cancer_type, *, regimen=None, fallback=True, inherit=True) -> dict:
    """Resolve the evidence source row used for an ICI response lookup.

    Returns lookup metadata without forcing callers to inspect the ORR map shape:

    - ``requested_cancer_code``: canonical code requested by the caller.
    - ``resolved_cancer_code``: direct/proxy/ancestor source row, when one exists.
    - ``inheritance_kind``: ``"direct"``, ``"source_scope"``, ``"ancestor"``,
      ``"direct_missing"``, or ``"missing"``.
    - ``available_regimens``: regimens available at the resolved source row.
    - source/provenance fields from the selected record when available.

    ``"direct_missing"`` is an *audited* gap: the code has a curated row stating that
    no single representative ORR describes it (with ``source_scope`` and
    ``missing_reason`` explaining why), so it reports
    ``has_ici_response_source=True`` and does not inherit an ancestor's value.
    ``"missing"`` remains the unreviewed case — no curated row of any kind.

    With ``regimen=None`` and ``fallback=True`` the selected record follows
    :data:`REGIMEN_FALLBACK`; with ``fallback=False`` the resolver still reports the
    resolved source and all available regimens but does not choose a single regimen.
    """
    requested_code = resolve_cancer_type(cancer_type)
    order = (regimen,) if regimen is not None else REGIMEN_FALLBACK
    resolved_code, inheritance_kind, rows = _resolve_ici_response_source(
        requested_code, order, inherit=inherit
    )
    if inheritance_kind == "direct_missing":
        return _gap_record(requested_code)
    selected_regimen = None
    selected_row = None
    if fallback or regimen is not None:
        for key in order:
            row = rows.get(str(key))
            if row is not None:
                selected_regimen = str(key)
                selected_row = row
                break
    record = (
        _record_from_row(
            selected_row,
            requested_code=requested_code,
            resolved_code=resolved_code,
            inheritance_kind=inheritance_kind,
        )
        if selected_row is not None
        else {}
    )
    record.update(
        {
            "requested_cancer_code": requested_code,
            "resolved_cancer_code": resolved_code if rows else None,
            "inheritance_kind": inheritance_kind,
            "is_inherited_evidence": bool(rows) and requested_code != resolved_code,
            "selected_regimen": selected_regimen,
            "available_regimens": tuple(rows),
            "has_ici_response_source": bool(rows),
        }
    )
    return record


def cancer_ici_response(
    cancer_type=None,
    *,
    regimen=None,
    fallback=True,
    inherit=True,
    include_inherited=False,
):
    """ICI objective response rate (%) for a cancer type.

    ``regimen`` pins one of :data:`REGIMEN_FALLBACK` (``"PD-1"`` / ``"PD-L1"`` /
    ``"PD-1+CTLA-4"``); leave it ``None`` to resolve across regimens.

    With ``regimen=None`` and ``fallback=True`` (default), the value is taken from the
    first regimen present in :data:`REGIMEN_FALLBACK` order (anti-PD-1 → anti-PD-L1 →
    combination) — the "best-available" anchor. With ``fallback=False`` the per-regimen
    mapping ``{regimen: orr}`` is returned instead.

    ``cancer_type`` is resolved via :func:`resolve_cancer_type`. With ``inherit``
    (default) a code with no row of its own inherits its nearest ancestor's value via
    the registry ``parent_code`` chain. Returns ``None`` (or ``{}``) when nothing is
    found.

    With ``cancer_type=None`` returns the whole direct ``{code: orr}`` map by default
    (a single regimen if pinned, else the fallback pick) — ready as a per-source
    plotting axis. Source-scoped children such as ``COAD_MSI`` and ``READ_MSI`` are not
    duplicated in default bulk maps. Pass ``include_inherited=True`` to expand across
    registry codes with the same resolver used for individual lookups.
    """
    maps = _regimen_maps()
    order = (regimen,) if regimen is not None else REGIMEN_FALLBACK

    if cancer_type is None:
        if include_inherited:
            out = {}
            for code in _bulk_lookup_codes(include_inherited=True):
                if regimen is not None:
                    value = cancer_ici_response(
                        code,
                        regimen=regimen,
                        inherit=inherit,
                    )
                    if value is not None:
                        out[code] = value
                elif not fallback:
                    values = cancer_ici_response(
                        code,
                        fallback=False,
                        inherit=inherit,
                    )
                    if values:
                        out[code] = values
                else:
                    value = cancer_ici_response(code, inherit=inherit)
                    if value is not None:
                        out[code] = value
            return out
        if regimen is not None:
            return dict(maps.get(regimen, {}))
        codes = {c for m in maps.values() for c in m}
        if not fallback:
            # Per-regimen mapping for every covered cancer: {code: {regimen: orr}}.
            return {c: {r: maps[r][c] for r in REGIMEN_FALLBACK if c in maps[r]} for c in codes}
        # Fallback pick per cancer across the union of covered codes.
        out = {}
        for c in codes:
            val, _ = _resolve_with_fallback(c, maps, REGIMEN_FALLBACK)
            if val is not None:
                out[c] = val
        return out

    code = resolve_cancer_type(cancer_type)

    if regimen is None and not fallback:
        per = {r: maps[r][code] for r in REGIMEN_FALLBACK if code in maps.get(r, {})}
        if per or not inherit:
            return per
        # Same audited-gap guard as the scalar path below: a reviewed gap must not
        # inherit an ancestor's per-regimen map either, or the two lookup paths would
        # disagree about whether the code has a value.
        if code in _gap_rows():
            return {}
        source_code = cancer_evidence_source_code(code)
        if source_code != code:
            hit = {
                r: maps[r][source_code] for r in REGIMEN_FALLBACK if source_code in maps.get(r, {})
            }
            if hit:
                return hit
        # walk ancestors for a per-regimen mapping
        reg = cancer_type_registry().set_index("code")
        cur, seen = code, set()
        while cur and cur not in seen:
            seen.add(cur)
            hit = {r: maps[r][cur] for r in REGIMEN_FALLBACK if cur in maps.get(r, {})}
            if hit:
                return hit
            cur = _parent_code(cur, reg)
        return {}

    val, _ = _resolve_with_fallback(code, maps, order)
    if val is not None or not inherit:
        return val
    # Keep the value path and the resolver in agreement: a curated gap row means this
    # code has been reviewed and has no representative ORR, so it must not inherit one.
    if code in _gap_rows():
        return None
    source_code = cancer_evidence_source_code(code)
    if source_code != code:
        val, _ = _resolve_with_fallback(source_code, maps, order)
        if val is not None:
            return val
    reg = cancer_type_registry().set_index("code")
    cur, seen = code, set()
    while cur and cur not in seen:
        seen.add(cur)
        val, _ = _resolve_with_fallback(cur, maps, order)
        if val is not None:
            return val
        cur = _parent_code(cur, reg)
    return None


def cancer_ici_response_record(
    cancer_type=None,
    *,
    regimen=None,
    fallback=True,
    inherit=True,
    include_inherited=False,
):
    """Metadata-bearing ICI objective response lookup.

    This mirrors :func:`cancer_ici_response`, but returns the resolved anchor row as a
    dict instead of only the ORR value. The record includes the joined evidence fields
    from :func:`cancer_ici_response_df` plus lookup metadata:

    - ``requested_cancer_code``: canonical code requested by the caller.
    - ``resolved_cancer_code``: source row used for the returned evidence.
    - ``inheritance_kind``: ``"direct"``, ``"source_scope"``, ``"ancestor"``, or
      ``"missing"``.
    - ``is_inherited_evidence``: whether the requested code differs from the resolved
      source row.

    Source-scoped biomarker cohorts such as ``COAD_MSI`` and ``READ_MSI`` therefore
    resolve through the curated ``CRC_MSI`` row while preserving the fact that the trial
    source is metastatic MSI-H/dMMR colorectal cancer rather than a colon- or
    rectum-specific estimate. With ``cancer_type=None`` the returned bulk map is direct
    source rows only by default; pass ``include_inherited=True`` to expand across
    registry codes with inherited source/proxy metadata.
    """
    order = (regimen,) if regimen is not None else REGIMEN_FALLBACK

    if cancer_type is None:
        codes = _bulk_lookup_codes(include_inherited=include_inherited)
        record_inherit = inherit if include_inherited else False
        if regimen is not None:
            return {
                code: record
                for code in codes
                if (
                    record := cancer_ici_response_record(
                        code, regimen=regimen, inherit=record_inherit
                    )
                )
            }
        if not fallback:
            return {
                code: records
                for code in codes
                if (
                    records := cancer_ici_response_record(
                        code, fallback=False, inherit=record_inherit
                    )
                )
            }
        return {
            code: record
            for code in codes
            if (record := cancer_ici_response_record(code, inherit=record_inherit))
        }

    requested_code = resolve_cancer_type(cancer_type)
    resolved_code, inheritance_kind, rows = _resolve_ici_response_rows(
        requested_code, order, inherit=inherit
    )

    if not fallback and regimen is None:
        # A gap names no regimen, so the per-regimen view is legitimately empty.
        return {
            key: _record_from_row(
                row,
                requested_code=requested_code,
                resolved_code=resolved_code,
                inheritance_kind=inheritance_kind,
            )
            for key, row in rows.items()
        }

    # Mirror tmb.cancer_tmb_record: an audited gap returns its curated record so the
    # record surface can distinguish a reviewed gap from an uncurated code. Only a
    # genuinely unresolved lookup returns None.
    if inheritance_kind == "direct_missing":
        return _gap_record(requested_code)

    for key in order:
        row = rows.get(str(key))
        if row is not None:
            return _record_from_row(
                row,
                requested_code=requested_code,
                resolved_code=resolved_code,
                inheritance_kind=inheritance_kind,
            )
    return None


def cancer_ici_regimen(cancer_type):
    """The regimen tag (``"PD-1"`` / ``"PD-L1"`` / ``"PD-1+CTLA-4"``) the fallback
    resolution selects for a cancer type — i.e. *which source* its
    :func:`cancer_ici_response` value comes from. Evidence-source fallback is applied
    for source-scoped rows such as ``COAD_MSI`` -> ``CRC_MSI``; parent-tree inheritance
    is not."""
    code = resolve_cancer_type(cancer_type)
    _, regimen = _resolve_with_fallback(code, _regimen_maps(), REGIMEN_FALLBACK)
    if regimen is None:
        source_code = cancer_evidence_source_code(code)
        if source_code != code:
            _, regimen = _resolve_with_fallback(source_code, _regimen_maps(), REGIMEN_FALLBACK)
    return regimen


# --------------------------------------------------------------------------------------
# Multi-endpoint estimates + pooling
#
# ``cancer-ici-response.csv`` carries ONE representative ORR anchor per (cancer, regimen).
# ``cancer-ici-response-estimates.csv`` is the wider evidence base behind it: every
# endpoint (ORR/CRR/DCR/DOR/PFS/OS + landmark rates) from every trial-source for that
# cell, with CIs and n, produced by the reference audit. The pooling helper combines
# those sources into a single responder-weighted estimate with a Wilson CI.
# --------------------------------------------------------------------------------------


def cancer_ici_response_estimates_df():
    """The verified ``cancer-ici-response-estimates.csv`` long table: one row per
    (``cancer_code``, ``regimen``, trial-source, ``metric``).

    Generalizes the one-anchor-per-cell :func:`cancer_ici_response_df` to *all* extracted
    endpoints — ORR, CRR, DCR, DOR, PFS, OS and landmark PFS/OS rates — each with
    ``value``, ``unit`` (``percent`` / ``months`` / ``rate_percent``), confidence interval
    (``ci_low`` / ``ci_high``), ``timepoint``, sample size (``metric_n`` / ``source_n``)
    and ``responders``. ``estimate_id`` is a stable row identifier used by compact
    anchor tables to point back to the exact supporting estimate. ``value_status``,
    ``ci_basis``, ``ci_low_status``, ``ci_high_status``, and
    ``source_locator_status`` make missing/not-reached/not-estimable provenance
    explicit. ``role`` is ``"primary"`` (the cited representative setting) or
    ``"alternate"`` (other trials / subgroups for the same cancer + regimen).

    ``source_verified`` preserves the prior evidence/citation verification decision;
    use ``source_locator_status="verified"`` when an exact public source block is
    required. ``value_basis`` distinguishes directly ``"reported"`` values from
    ``"computed_from_counts"``, ``"inferred_from_outcomes"``,
    ``"derived_cross_cohort"``, ``"reported_context"``, and ``"derived_blend"``.
    Pooling excludes inferred, cross-cohort, contextual, and blended values."""
    return get_data("cancer-ici-response-estimates")


def cancer_ici_source_locator_audit_df():
    """The row-level source audit behind :func:`cancer_ici_response_estimates_df`.

    Each ``estimate_id`` appears exactly once with the public source URL and document
    kind, the table/figure/section locator available during the audit, match evidence,
    and audit date. A ``citation_only`` status means the citation was confirmed but no
    public text block was available; it does not claim an invented article location.
    """
    return get_data("cancer-ici-source-locator-audit")


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _wilson_ci(k: float, n: float, z: float = 1.96):
    """95% Wilson score interval (returned as percentages) for ``k`` responders of ``n``."""
    if not n:
        return (None, None)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(100 * (center - half), 1), round(100 * (center + half), 1))


def pooled_ici_response(
    cancer_type,
    *,
    regimen=None,
    metric="ORR",
    verified_only=True,
    include_alternates=True,
):
    """Pool every audited estimate for one cancer + regimen + endpoint.

    Returns a dict::

        {cancer_code, regimen, metric, poolable, pooled_pct, ci_low, ci_high,
         responders_total, n_total, n_studies, n_pooled, refs, value_range, sources}

    For **proportion endpoints** (:data:`PROPORTION_METRICS` — ORR / CRR / DCR / PR) the
    pool is *responder-weighted*: ``pooled_pct = 100 · Σresponders / Σn`` over the sources
    that report both, with a 95% Wilson score CI. ``n_total`` is the summed sample size and
    ``responders_total`` the summed responders. ``n_studies`` is the number of trial-sources
    found for the cell (the full evidence count); ``n_pooled`` is how many of them actually
    entered the responder-weighted pool (``None`` for non-proportion endpoints). ``refs``
    lists the citations behind the reported estimate.

    For **time-to-event endpoints** (median PFS/OS/DOR in months) and **landmark rates**,
    medians/rates cannot be validly pooled without patient-level data — ``poolable`` is
    ``False``, ``pooled_pct`` is ``None``, and only the per-trial ``sources`` and their
    ``value_range`` are returned.

    Setting heterogeneity is real (all-comer vs PD-L1/MSI-selected vs different lines).
    By default the pool includes the cited primary setting *and* the ``alternate`` rows;
    pass ``include_alternates=False`` to pool only the representative primary setting, or
    inspect each source's ``setting`` in ``sources`` to judge comparability. The
    per-source breakdown and ``value_range`` are always returned so heterogeneity (and
    any overlapping subgroups) stays visible. ``verified_only`` (default) keeps only
    audit-confirmed citations.
    """
    requested_code = resolve_cancer_type(cancer_type)
    code = cancer_evidence_source_code(requested_code)
    metric = str(metric).upper()
    df = cancer_ici_response_estimates_df()
    sub = df[(df["cancer_code"] == code) & (df["metric"].astype(str).str.upper() == metric)]
    if regimen is not None:
        sub = sub[sub["regimen"] == regimen]
    # Derived blends and contextual comparator/overlapping rows are audit evidence, not
    # subtype-specific pool inputs.
    if "value_basis" in sub.columns:
        basis = sub["value_basis"].astype(str)
        sub = sub[~basis.isin(NON_POOLABLE_VALUE_BASIS)]
    if verified_only:
        sub = sub[sub["source_verified"].map(_truthy)]
    if not include_alternates:
        sub = sub[sub["role"] == "primary"]

    def _num(v):
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    sources, seen = [], set()
    for _, r in sub.iterrows():
        ref = None if r.get("ref") is None else str(r.get("ref"))
        dedupe = (ref, str(r.get("trial_name")), str(r.get("setting")), _num(r.get("value")))
        if dedupe in seen:
            continue
        seen.add(dedupe)
        sources.append(
            {
                "role": r.get("role"),
                "drug": r.get("drug"),
                "trial_name": r.get("trial_name"),
                "trial_alias": r.get("trial_alias"),
                "trial_nct": r.get("trial_nct"),
                "ref": ref,
                "setting": r.get("setting"),
                "value": _num(r.get("value")),
                "unit": r.get("unit"),
                "ci_low": _num(r.get("ci_low")),
                "ci_high": _num(r.get("ci_high")),
                "n": _num(r.get("metric_n")),
                "responders": _num(r.get("responders")),
            }
        )

    values = [s["value"] for s in sources if s["value"] is not None]
    # Sources that actually enter the responder-weighted pool (need responders + n);
    # empty for non-proportion endpoints, which are not pooled.
    contrib = (
        [s for s in sources if s["responders"] is not None and s["n"]]
        if metric in PROPORTION_METRICS
        else []
    )
    # `refs` = the citations behind the *reported* estimate: the pooled sources when a
    # pool is produced, else every source feeding the value_range.
    ref_sources = contrib if contrib else sources
    result = {
        "cancer_code": code,
        "requested_cancer_code": requested_code,
        "regimen": regimen,
        "metric": metric,
        "poolable": metric in PROPORTION_METRICS,
        "pooled_pct": None,
        "ci_low": None,
        "ci_high": None,
        "responders_total": None,
        "n_total": None,
        "n_studies": len(sources),  # trial-sources found for this cell + metric
        "n_pooled": len(contrib) if metric in PROPORTION_METRICS else None,  # entered the pool
        "refs": sorted({s["ref"] for s in ref_sources if s["ref"]}),
        "value_range": (min(values), max(values)) if values else None,
        "sources": sources,
    }

    if contrib:
        k = sum(s["responders"] for s in contrib)
        n = sum(s["n"] for s in contrib)
        if n:
            lo, hi = _wilson_ci(k, n)
            result.update(
                pooled_pct=round(100 * k / n, 1),
                ci_low=lo,
                ci_high=hi,
                responders_total=int(k),
                n_total=int(n),
            )
    return result
