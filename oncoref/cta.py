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

"""Cancer-testis antigens (CTAs): the gene list + HPA tissue-restriction evidence.

A CTA is a gene whose normal expression is restricted to reproductive tissues
(testis / ovary / placenta) and which reactivates in tumors — a tissue-restriction
call over HPA normal-tissue expression, i.e. cancer **reference data**. oncoref
owns the definition: the bundled ``cancer-testis-antigens.csv`` carries the
candidate list (from 5 source databases) plus the HPA-derived per-tissue
restriction columns and filter flags.

The MS-evidence restriction tiers and peptide/MHC presentation that build on top
of this list are the target-selection layer's domain and are intentionally NOT
here. ``restriction`` and ``restriction_confidence`` in the bundled table are the
**HPA-only** synthesis (protein + RNA modalities; see :func:`synthesize_restriction`)
— no MS contribution — so the values match the data oncoref owns.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

import pandas as pd

from .cta_tissues import HPA_EXPRESSION_FLOOR_NTPM
from .load_dataset import _register_derived_cache, get_data


def _never_expressed_rescue_mask(df: pd.DataFrame) -> pd.Series:
    """Row mask for ``never_expressed`` CTAs kept in the expressed set anyway.

    A uniform rule, not a per-gene list: a never-expressed gene is rescued when its
    HPA evidence is ``restriction_confidence == MODERATE`` and
    ``rna_restriction_level == STRICT`` — i.e. it has reproductive-restricted RNA
    just below the protein floor, the signature of a borderline-but-real CTA
    (testis ~1-2 nTPM). This replaces the old one-gene XAGE5 override, which
    rescued a single gene while ~15 peers with equal/stronger signal were dropped;
    the rule keeps XAGE5 and all of them on the same principled basis.
    """
    if not {"restriction_confidence", "rna_restriction_level"} <= set(df.columns):
        return pd.Series(False, index=df.index)
    moderate = df["restriction_confidence"].astype(str).str.upper() == "MODERATE"
    strict = df["rna_restriction_level"].astype(str).str.upper() == "STRICT"
    return moderate & strict


def _alpha_tubulin_symbol(symbol: str) -> bool:
    """Alpha-tubulin family (``TUBA1A``, ``TUBA3C``, …) — ubiquitous structural
    housekeeping genes, not tumor-restricted antigens."""
    return bool(re.match(r"^TUBA\d", str(symbol)))


@lru_cache(maxsize=1)
def _non_cta_excluded_gene_ids() -> frozenset[str]:
    """Unversioned Ensembl IDs excluded from the CTA universe by a gene-family rule:
    a candidate that is a **core histone** (member of ``histone-genes.csv``) or an
    **alpha-tubulin** is a ubiquitous structural housekeeping gene that entered via a
    source database but is not a tumor-restricted antigen.

    Deriving the set from gene family (rather than a hand-listed set of IDs) keeps it
    self-maintaining and consistent: it caught H1-6, a core histone that passed the
    HPA filter exactly like its deny-listed siblings H2BC1/H1-1 but had been left in
    (the same one-gene inconsistency fixed for CGB8 in #20). The placental hCG-beta
    locus CGB8 is **not** here — it is a real reproductive-restricted antigen like
    CGB1/2/3/5/7, not a housekeeping gene.
    """
    from .gene_families import gene_family_ids

    df = get_data("cancer-testis-antigens", copy=False)
    if "Ensembl_Gene_ID" not in df.columns:
        return frozenset()
    unversioned = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    histones = unversioned.isin(gene_family_ids("histone"))
    tubulins = df["Symbol"].map(_alpha_tubulin_symbol) if "Symbol" in df.columns else False
    return frozenset(unversioned[histones | tubulins])


_register_derived_cache(_non_cta_excluded_gene_ids.cache_clear)

#: Backwards-compatible snapshot of the family-derived exclusion set (see
#: :func:`_non_cta_excluded_gene_ids`). Computed once at import; the live filter in
#: :func:`_cta_frame` calls the cached function so a fixture swap stays correct.
NON_CTA_EXCLUDED_GENE_IDS: frozenset[str] = _non_cta_excluded_gene_ids()

_PASSES_FILTERS_COLUMN = "passes_filters"
_LEGACY_FILTERED_COLUMN = "filtered"
_NO_PROTEIN_RELIABILITY = {"no data", "nan", ""}
_REPRODUCTIVE_RESTRICTION_LABELS = frozenset({"TESTIS", "PLACENTAL", "REPRODUCTIVE"})
_CANONICAL_ALIAS_OVERRIDES: dict[str, str] = {
    "NYESO1": "CTAG1B",
    "ESO1": "CTAG1B",
}


def _normalize_alias(name: object) -> str:
    """Case- and punctuation-insensitive key for CTA symbols and aliases."""
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def _rna_restriction_agrees(selected_restriction: str, rna_restriction: str) -> bool:
    """Whether an RNA call supports the selected HPA restriction.

    Exact calls agree. The broader RNA label ``REPRODUCTIVE`` also supports a
    more specific reproductive protein call, but never a ``SOMATIC`` call.
    """
    if rna_restriction == selected_restriction:
        return True
    return (
        rna_restriction == "REPRODUCTIVE"
        and selected_restriction in _REPRODUCTIVE_RESTRICTION_LABELS
    )


def synthesize_restriction(row) -> tuple[str, str]:
    """HPA-only tissue restriction + confidence for a CTA row.

    Best tissue modality (protein > RNA); ``restriction_confidence`` is HIGH /
    MODERATE / LOW / NO_DATA from per-modality agreement and HPA reliability.
    This is the **HPA-only** synthesis — the MS-evidence contribution that the
    target-selection layer adds is intentionally excluded, so the value's
    provenance matches the data oncoref owns.
    """
    protein_r = str(row.get("protein_restriction", "") or "")
    protein_rel = str(row.get("protein_reliability", "") or "")
    rna_r = str(row.get("rna_restriction", "") or "")
    rna_level = str(row.get("rna_restriction_level", "") or "")

    protein_has_data = bool(protein_r) and protein_r != "NO_DATA"
    rna_has_data = bool(rna_r) and rna_r != "NO_DATA"
    if protein_has_data:
        tissue = protein_r
    elif rna_has_data:
        tissue = rna_r
    else:
        tissue = "NO_DATA"

    score = 0.0
    sources = 0
    if protein_has_data:
        sources += 1
        score += 1.0
        if protein_rel in ("Enhanced", "Supported"):
            score += 0.5
    if rna_has_data:
        sources += 1
        if _rna_restriction_agrees(tissue, rna_r):
            score += 1.0
            if rna_level == "STRICT":
                score += 0.5

    if sources == 0:
        confidence = "NO_DATA"
    elif score / sources >= 1.2:
        confidence = "HIGH"
    elif score / sources >= 0.8:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    # Cap HIGH when the only evidence is RNA below the expression floor: the scorer
    # credits any STRICT RNA equally, so a gene at ~1-2 nTPM (never_expressed, no
    # protein) would otherwise earn HIGH from near-noise RNA. Genes with protein
    # evidence are untouched. (HPA-only port of tsarina#114; no MS term here.)
    if confidence == "HIGH" and not protein_has_data:
        rna_max = pd.to_numeric(row.get("rna_max_ntpm"), errors="coerce")
        if pd.notna(rna_max) and rna_max < HPA_EXPRESSION_FLOOR_NTPM:
            confidence = "MODERATE"
    return tissue, confidence


@lru_cache(maxsize=1)
def _cta_frame() -> pd.DataFrame:
    """Cached CTA table with the non-CTA excluded genes dropped. Internal,
    read-only — do not mutate; public callers get a copy via cta_df()."""
    df = get_data("cancer-testis-antigens", copy=False)
    excluded = _non_cta_excluded_gene_ids()
    if "Ensembl_Gene_ID" in df.columns and excluded:
        unversioned = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
        df = df[~unversioned.isin(excluded)].reset_index(drop=True)
    return df


_register_derived_cache(_cta_frame.cache_clear)


def cta_df() -> pd.DataFrame:
    """Full CTA evidence table (one row per candidate), with the non-CTA
    excluded genes (histones, etc.) dropped. Returns a defensive copy."""
    return _with_specificity_columns(_cta_frame())


#: Alias matching the target-selection layer's public name.
def cta_evidence() -> pd.DataFrame:
    """The CTA evidence DataFrame (alias of :func:`cta_df`)."""
    return cta_df()


def _specificity_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    passes = passes_filters_mask(df)
    never = _boolish(df["never_expressed"]) if "never_expressed" in df.columns else False
    rescued = _never_expressed_rescue_mask(df)
    default_included = passes & ~(never & ~rescued)
    out["specificity_status"] = "excluded_normal_expression"
    out.loc[passes & ~default_included, "specificity_status"] = "canonical_low_expression"
    out.loc[default_included, "specificity_status"] = "canonical_default"
    out["specificity_action"] = "exclude_default"
    out.loc[default_included, "specificity_action"] = "include_default"
    out["specificity_source_anchor"] = "derived:oncoref.cta.passes_filters"
    out["specificity_rationale"] = (
        "Derived from HPA reproductive-restriction filters and never-expressed rescue policy."
    )
    return out


def _with_specificity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Ensembl_Gene_ID" not in out.columns:
        return out
    defaults = _specificity_defaults(out)
    for col in defaults.columns:
        out[col] = defaults[col].to_numpy()

    audit = cta_specificity_audit_references()
    if audit.empty:
        return out
    audit = audit.copy()
    audit["Ensembl_Gene_ID"] = audit["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    audit_cols = [
        "Ensembl_Gene_ID",
        "specificity_status",
        "specificity_action",
        "source_anchor",
        "rationale",
    ]
    joined = out.assign(
        Ensembl_Gene_ID=out["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    ).merge(
        audit[audit_cols].rename(
            columns={
                "source_anchor": "specificity_source_anchor",
                "rationale": "specificity_rationale",
            }
        ),
        on="Ensembl_Gene_ID",
        how="left",
        suffixes=("", "_audit"),
    )
    for col in ("specificity_status", "specificity_action"):
        audit_col = f"{col}_audit"
        out[col] = joined[audit_col].where(joined[audit_col].notna(), out[col]).to_numpy()
    for col in ("specificity_source_anchor", "specificity_rationale"):
        audit_col = f"{col}_audit"
        out[col] = joined[audit_col].where(joined[audit_col].notna(), out[col]).to_numpy()
    return out


@lru_cache(maxsize=1)
def _alias_to_symbol() -> dict[str, str]:
    """Normalized alias/synonym -> official CTA symbol.

    Precedence follows the target-selection API this surface is replacing:
    curated colloquial-name overrides, then official symbols, then first table
    alias in row order.
    """
    df = _cta_frame()
    symbols = set(df["Symbol"].astype(str)) if "Symbol" in df.columns else set()
    mapping: dict[str, str] = {}
    if "Aliases" in df.columns:
        for symbol, aliases in zip(df["Symbol"], df["Aliases"]):
            if not isinstance(aliases, str):
                continue
            for alias in aliases.split(";"):
                key = _normalize_alias(alias)
                if key and key not in mapping:
                    mapping[key] = str(symbol)
    if "Symbol" in df.columns:
        for symbol in df["Symbol"]:
            key = _normalize_alias(symbol)
            if key:
                mapping[key] = str(symbol)
    for key, symbol in _CANONICAL_ALIAS_OVERRIDES.items():
        if symbol in symbols:
            mapping[key] = symbol
    return mapping


_register_derived_cache(_alias_to_symbol.cache_clear)


def cta_symbol_for_alias(name: str) -> str | None:
    """Resolve a CTA symbol/synonym to the official table symbol.

    Punctuation and case are ignored, so ``"NY-ESO-1"``, ``"ESO1"``, and
    ``"ny eso 1"`` all resolve to ``"CTAG1B"``. Unknown names return ``None``.
    """
    return _alias_to_symbol().get(_normalize_alias(name))


def cta_candidate_references() -> pd.DataFrame:
    """Top-of-funnel CTA *candidates* with literature references — overlooked
    cancer-testis / cancer-germline antigens (paralog-family members, meiosis/
    germline genes, recently described CTAs) that are **not yet promoted** into the
    curated :func:`cta_df` table.

    This is a referenced watchlist, not the filtered set: each row carries its
    ``candidate_source`` (paralog/literature/meiosis/registry), ``ct_designation``
    (CTdatabase CT-number where one exists), ``pmids`` (semicolon-joined), the
    ``family`` it belongs to, and the HPA v23 status (``hpa_testis_ntpm``,
    ``hpa_max_somatic_ntpm``, ``hpa_max_somatic_tissue``, ``hpa_testis_restricted``)
    so a curator can see at a glance which are clean testis-restricted promotions
    vs. which carry somatic signal that needs a cross-reactivity/leakiness call
    before entering the curated table. Returns a defensive copy."""
    return get_data("cta-candidate-references").copy()


def cta_specificity_audit_references() -> pd.DataFrame:
    """Machine-readable specificity audit decisions for flagged CTA candidates.

    This is the source table for explicit demotion/candidate-only calls. It is
    intentionally narrower than :func:`cta_df`: rows appear here when a gene needs
    a curator-visible specificity decision beyond the derived HPA filter status.
    """
    return get_data("cta-specificity-audit").copy()


def cta_specificity_audit() -> pd.DataFrame:
    """Specificity audit rows joined to HPA/candidate evidence.

    The raw audit table records the decision, action, source anchor, and rationale.
    This helper adds the normal-tissue columns needed to review or reproduce the
    decision without searching the broader CTA and candidate tables.
    """
    audit = cta_specificity_audit_references()
    audit["Ensembl_Gene_ID"] = audit["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]

    table_cols = [
        "Symbol",
        "Ensembl_Gene_ID",
        "passes_filters",
        "never_expressed",
        "rna_testis_ntpm",
        "rna_ovary_ntpm",
        "rna_placenta_ntpm",
        "rna_max_somatic_tissue",
        "rna_max_somatic_ntpm",
        "rna_somatic_detected_count",
        "rna_brain_max_ntpm",
        "rna_heart_max_ntpm",
        "protein_restriction",
        "rna_restriction",
        "rna_restriction_level",
        "restriction",
        "restriction_confidence",
        "safety_flags",
    ]
    table = _cta_frame()[table_cols].copy()
    table["Ensembl_Gene_ID"] = table["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    out = audit.merge(
        table.drop(columns=["Symbol"]).rename(columns={"passes_filters": "cta_passes_filters"}),
        on="Ensembl_Gene_ID",
        how="left",
    )

    candidates = cta_candidate_references().copy()
    candidates["Ensembl_Gene_ID"] = candidates["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    candidate_cols = [
        "Ensembl_Gene_ID",
        "candidate_source",
        "ct_designation",
        "family",
        "hpa_testis_ntpm",
        "hpa_max_somatic_ntpm",
        "hpa_max_somatic_tissue",
        "hpa_testis_restricted",
    ]
    out = out.merge(candidates[candidate_cols], on="Ensembl_Gene_ID", how="left")
    out["in_cta_table"] = out["cta_passes_filters"].notna()
    out["in_candidate_watchlist"] = out["candidate_source"].notna()
    return out.copy()


def cta_clinical_target_references() -> pd.DataFrame:
    """Source-anchored clinical/canonical CTA target tier.

    This table is deliberately separate from :func:`cta_filtered_gene_names`: the
    strict default remains the HPA reproductive-restriction view, while this tier
    keeps well-known clinical or registry CTAs discoverable even when they fail the
    strict normal-tissue screen or have not yet been promoted out of the candidate
    watchlist.
    """
    return get_data("cta-clinical-targets").copy()


def _boolish(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower() == "true"


def _clinical_evidence_tier(row: pd.Series) -> str:
    if pd.notna(row.get("passes_filters")):
        if bool(row.get("_passes_filters")):
            return (
                "filtered_never_expressed"
                if bool(row.get("_never_expressed"))
                else "filtered_expressed"
            )
        return "excluded"
    if pd.notna(row.get("candidate_source")):
        return "candidate"
    return "clinical_only"


def cta_clinical_target_evidence() -> pd.DataFrame:
    """Clinical/canonical CTA target tier joined to HPA and candidate evidence.

    The joined ``evidence_tier`` makes the safety boundary explicit:

    - ``filtered_expressed`` / ``filtered_never_expressed``: already in the strict
      oncoref CTA table;
    - ``excluded``: in the CTA table but failed the HPA filter, with
      ``exclusion_driver_tissue`` / ``exclusion_driver_ntpm`` exposing the somatic
      RNA signal that drove exclusion;
    - ``candidate``: in the referenced candidate watchlist but not the curated CTA
      table;
    - ``clinical_only``: source-anchored target row without bundled HPA evidence.

    This helper is for explicit clinical/canonical CTA workflows. It does not alter
    the default strict CTA helpers.
    """
    refs = cta_clinical_target_references()
    refs["Ensembl_Gene_ID"] = refs["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]

    table_cols = [
        "Symbol",
        "Ensembl_Gene_ID",
        "source_databases",
        "passes_filters",
        "never_expressed",
        "rna_testis_ntpm",
        "rna_ovary_ntpm",
        "rna_placenta_ntpm",
        "rna_max_somatic_tissue",
        "rna_max_somatic_ntpm",
        "rna_somatic_detected_count",
        "rna_brain_max_ntpm",
        "rna_heart_max_ntpm",
        "protein_restriction",
        "rna_restriction",
        "rna_restriction_level",
        "restriction",
        "restriction_confidence",
        "safety_flags",
    ]
    table = cta_df()[table_cols].copy()
    table["Ensembl_Gene_ID"] = table["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    table["_passes_filters"] = passes_filters_mask(table)
    table["_never_expressed"] = _boolish(table["never_expressed"])

    out = refs.merge(
        table.drop(columns=["Symbol"]).rename(columns={"source_databases": "cta_source_databases"}),
        on="Ensembl_Gene_ID",
        how="left",
    )

    candidates = cta_candidate_references().copy()
    candidates["Ensembl_Gene_ID"] = candidates["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    candidate_cols = [
        "Ensembl_Gene_ID",
        "candidate_source",
        "ct_designation",
        "family",
        "hpa_testis_ntpm",
        "hpa_max_somatic_ntpm",
        "hpa_max_somatic_tissue",
        "hpa_testis_restricted",
        "pmids",
    ]
    out = out.merge(
        candidates[candidate_cols].rename(columns={"pmids": "candidate_pmids"}),
        on="Ensembl_Gene_ID",
        how="left",
    )
    out["evidence_tier"] = out.apply(_clinical_evidence_tier, axis=1)
    excluded = out["evidence_tier"] == "excluded"
    out["exclusion_driver_tissue"] = out["rna_max_somatic_tissue"].where(excluded)
    out["exclusion_driver_ntpm"] = pd.to_numeric(
        out["rna_max_somatic_ntpm"], errors="coerce"
    ).where(excluded)
    out = out.drop(columns=["_passes_filters", "_never_expressed"])

    cols = [
        "Symbol",
        "Ensembl_Gene_ID",
        "clinical_tier",
        "evidence_tier",
        "source_anchor",
        "pmids",
        "rationale",
        "passes_filters",
        "never_expressed",
        "exclusion_driver_tissue",
        "exclusion_driver_ntpm",
        "rna_testis_ntpm",
        "rna_ovary_ntpm",
        "rna_placenta_ntpm",
        "rna_max_somatic_tissue",
        "rna_max_somatic_ntpm",
        "rna_somatic_detected_count",
        "rna_brain_max_ntpm",
        "rna_heart_max_ntpm",
        "protein_restriction",
        "rna_restriction",
        "rna_restriction_level",
        "restriction",
        "restriction_confidence",
        "safety_flags",
        "cta_source_databases",
        "candidate_source",
        "ct_designation",
        "family",
        "hpa_testis_ntpm",
        "hpa_max_somatic_ntpm",
        "hpa_max_somatic_tissue",
        "hpa_testis_restricted",
        "candidate_pmids",
    ]
    return out[cols].copy()


def cta_clinical_target_gene_names() -> set[str]:
    """Symbols in the explicit clinical/canonical CTA target tier."""
    return set(cta_clinical_target_references()["Symbol"].astype(str))


def cta_clinical_target_gene_ids() -> set[str]:
    """Ensembl IDs in the explicit clinical/canonical CTA target tier."""
    return set(
        cta_clinical_target_references()["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    )


def cta_excluded_clinical_target_gene_names() -> set[str]:
    """Clinical/canonical CTA targets present in the CTA table but excluded by the
    strict HPA filter."""
    df = cta_clinical_target_evidence()
    return set(df.loc[df["evidence_tier"] == "excluded", "Symbol"].astype(str))


def cta_excluded_clinical_target_gene_ids() -> set[str]:
    """Ensembl IDs for :func:`cta_excluded_clinical_target_gene_names`."""
    df = cta_clinical_target_evidence()
    return set(df.loc[df["evidence_tier"] == "excluded", "Ensembl_Gene_ID"].astype(str))


def passes_filters_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask for rows passing the HPA curation filter (reproductive
    restriction). Accepts the legacy ``filtered`` column too."""
    if _PASSES_FILTERS_COLUMN in df.columns:
        values = df[_PASSES_FILTERS_COLUMN]
    elif _LEGACY_FILTERED_COLUMN in df.columns:
        values = df[_LEGACY_FILTERED_COLUMN]
    else:
        return pd.Series(True, index=df.index)
    return values.astype(str).str.lower() == "true"


def _cta_with_specificity_frame() -> pd.DataFrame:
    """CTA table with machine-readable specificity decisions joined in."""
    return _with_specificity_columns(_cta_frame())


def _canonical_filtered_mask(df: pd.DataFrame) -> pd.Series:
    """Rows in the canonical CTA specificity tier, including low-expression CTAs.

    ``passes_filters`` is the raw HPA reproductive-restriction filter. Public CTA
    set helpers use the explicit specificity decision on top of that raw filter,
    so audited demotions are excluded from defaults even if they otherwise pass
    the mechanical HPA gate.
    """
    if {"specificity_action", "specificity_status"} <= set(df.columns):
        action = df["specificity_action"].astype(str)
        status = df["specificity_status"].astype(str)
        return action.eq("include_default") | status.eq("canonical_low_expression")
    return passes_filters_mask(df)


def _canonical_default_mask(df: pd.DataFrame) -> pd.Series:
    """Rows in the recommended expressed canonical CTA default set."""
    if "specificity_action" in df.columns:
        return df["specificity_action"].astype(str).eq("include_default")
    mask = passes_filters_mask(df)
    if "never_expressed" in df.columns:
        never = df["never_expressed"].astype(str).str.lower() == "true"
        never = never & ~_never_expressed_rescue_mask(df)
        mask = mask & ~never
    return mask


def _cta_by_column(
    column: str,
    *,
    filtered_only: bool = False,
    exclude_never_expressed: bool = False,
) -> set[str]:
    df = _cta_with_specificity_frame() if filtered_only else _cta_frame()
    mask = pd.Series(True, index=df.index)
    if filtered_only and exclude_never_expressed:
        mask = _canonical_default_mask(df)
    elif filtered_only:
        mask = _canonical_filtered_mask(df)
    elif exclude_never_expressed and "never_expressed" in df.columns:
        never = df["never_expressed"].astype(str).str.lower() == "true"
        never = never & ~_never_expressed_rescue_mask(df)
        mask = mask & ~never
    subset = df[mask]
    result: set[str] = set()
    if column in subset.columns:
        for x in subset[column]:
            if isinstance(x, str):
                result.update(xi.strip() for xi in x.split(";"))
    return result


def _all_by_column(column: str) -> set[str]:
    df = _cta_frame()
    result: set[str] = set()
    if column in df.columns:
        for x in df[column]:
            if isinstance(x, str):
                result.update(xi.strip() for xi in x.split(";"))
    return result


def cta_gene_names() -> set[str]:
    """Canonical default CTA gene symbols: genes with explicit specificity action
    ``include_default`` after HPA filtering, never-expressed rescue, and audited
    specificity demotions."""
    return _cta_by_column("Symbol", filtered_only=True, exclude_never_expressed=True)


def cta_gene_ids() -> set[str]:
    """Canonical default CTA Ensembl gene IDs."""
    return _cta_by_column("Ensembl_Gene_ID", filtered_only=True, exclude_never_expressed=True)


def cta_filtered_gene_names() -> set[str]:
    """Canonical filtered CTA symbols, including low-expression CTAs.

    This is stricter than the raw ``passes_filters`` column when a row has an
    explicit specificity-audit demotion.
    """
    return _cta_by_column("Symbol", filtered_only=True)


def cta_filtered_gene_ids() -> set[str]:
    """Canonical filtered CTA Ensembl gene IDs, including low-expression CTAs."""
    return _cta_by_column("Ensembl_Gene_ID", filtered_only=True)


def cta_never_expressed_gene_names() -> set[str]:
    """Filter-passing CTAs with no meaningful HPA expression (no protein, max RNA < 2)."""
    return cta_filtered_gene_names() - cta_gene_names()


def cta_never_expressed_gene_ids() -> set[str]:
    """Filter-passing CTA Ensembl IDs with no meaningful HPA expression."""
    return cta_filtered_gene_ids() - cta_gene_ids()


def cta_unfiltered_gene_names() -> set[str]:
    """Every candidate CTA symbol across all source databases (the full universe)."""
    return _all_by_column("Symbol")


def cta_unfiltered_gene_ids() -> set[str]:
    """Every candidate CTA Ensembl gene ID across all source databases."""
    return _all_by_column("Ensembl_Gene_ID")


def cta_excluded_gene_names() -> set[str]:
    """Candidate CTAs outside the canonical specificity tier."""
    return cta_unfiltered_gene_names() - cta_filtered_gene_names()


def cta_excluded_gene_ids() -> set[str]:
    """Candidate CTA Ensembl IDs outside the canonical specificity tier."""
    return cta_unfiltered_gene_ids() - cta_filtered_gene_ids()


def _relaxed_reproductive_mask(df: pd.DataFrame, min_deflated_frac: float) -> pd.Series:
    failed = ~passes_filters_mask(df)
    rna_only = (
        df["protein_reliability"].astype(str).str.strip().str.lower().isin(_NO_PROTEIN_RELIABILITY)
    )
    frac = pd.to_numeric(df["rna_deflated_reproductive_frac"], errors="coerce")
    return failed & rna_only & (frac >= float(min_deflated_frac))


def cta_relaxed_reproductive_gene_names(min_deflated_frac: float = 0.80) -> set[str]:
    """Opt-in relaxed tier of RNA-only reproductive-dominant candidate CTAs.

    These candidates fail the default reproductive-restriction gate but have no
    HPA protein evidence and retain a high deflated reproductive RNA fraction.
    The tier is intentionally disjoint from :func:`cta_filtered_gene_names`.
    """
    df = _cta_frame()
    return set(df.loc[_relaxed_reproductive_mask(df, min_deflated_frac), "Symbol"])


def cta_relaxed_reproductive_gene_ids(min_deflated_frac: float = 0.80) -> set[str]:
    """Ensembl IDs for :func:`cta_relaxed_reproductive_gene_names`."""
    df = _cta_frame()
    return set(df.loc[_relaxed_reproductive_mask(df, min_deflated_frac), "Ensembl_Gene_ID"])


def _filter_values(values: str | Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = {values}
    return {str(v).upper() for v in values}


def cta_by_axes(
    *,
    restriction: str | Iterable[str] | None = None,
    protein_restriction: str | Iterable[str] | None = None,
    rna_restriction: str | Iterable[str] | None = None,
    rna_restriction_level: str | Iterable[str] | None = None,
    ms_restriction: str | Iterable[str] | None = None,
    restriction_confidence: str | Iterable[str] | None = None,
    column: str = "Symbol",
    filtered_only: bool = True,
) -> set[str]:
    """Return CTA identifiers matching restriction-axis filters.

    ``ms_restriction`` is accepted for target-selection API compatibility, but
    oncoref's base CTA table is HPA-only. If an MS filter is requested against a
    table with no MS column, the result is deliberately empty rather than an
    accidentally unfiltered set.
    """
    df = _cta_with_specificity_frame()
    if column not in df.columns:
        return set()
    mask = _canonical_filtered_mask(df) if filtered_only else pd.Series(True, index=df.index)
    for axis_col, values in (
        ("restriction", restriction),
        ("protein_restriction", protein_restriction),
        ("rna_restriction", rna_restriction),
        ("rna_restriction_level", rna_restriction_level),
        ("ms_restriction", ms_restriction),
        ("restriction_confidence", restriction_confidence),
    ):
        wanted = _filter_values(values)
        if wanted is None:
            continue
        if axis_col not in df.columns:
            return set()
        actual = df[axis_col].astype(str).str.upper()
        mask = mask & actual.isin(wanted)
    return _extract_values(df[mask], column)


def _extract_values(df: pd.DataFrame, column: str) -> set[str]:
    result: set[str] = set()
    if column in df.columns:
        for x in df[column]:
            if isinstance(x, str):
                result.update(xi.strip() for xi in x.split(";") if xi.strip())
    return result


def cta_testis_restricted_gene_names() -> set[str]:
    """Filter-passing CTAs with synthesized HPA restriction ``TESTIS``."""
    return cta_by_axes(restriction="TESTIS")


def cta_testis_restricted_gene_ids() -> set[str]:
    """Ensembl IDs for testis-restricted CTAs."""
    return cta_by_axes(restriction="TESTIS", column="Ensembl_Gene_ID")


def cta_placental_restricted_gene_names() -> set[str]:
    """Filter-passing CTAs with synthesized HPA restriction ``PLACENTAL``."""
    return cta_by_axes(restriction="PLACENTAL")


def cta_placental_restricted_gene_ids() -> set[str]:
    """Ensembl IDs for placental-restricted CTAs."""
    return cta_by_axes(restriction="PLACENTAL", column="Ensembl_Gene_ID")


def cta_gene_id_to_name() -> dict[str, str]:
    """``{Ensembl_Gene_ID (unversioned): Symbol}`` over the expressed CTA set."""
    df = _cta_frame()
    ids = cta_gene_ids()
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        gid = str(row.get("Ensembl_Gene_ID", "")).split(".")[0]
        if gid in ids:
            out[gid] = str(row.get("Symbol", gid))
    return out


CTA_gene_names = cta_gene_names
CTA_gene_ids = cta_gene_ids
CTA_filtered_gene_names = cta_filtered_gene_names
CTA_filtered_gene_ids = cta_filtered_gene_ids
CTA_never_expressed_gene_names = cta_never_expressed_gene_names
CTA_never_expressed_gene_ids = cta_never_expressed_gene_ids
CTA_unfiltered_gene_names = cta_unfiltered_gene_names
CTA_unfiltered_gene_ids = cta_unfiltered_gene_ids
CTA_excluded_gene_names = cta_excluded_gene_names
CTA_excluded_gene_ids = cta_excluded_gene_ids
CTA_testis_restricted_gene_names = cta_testis_restricted_gene_names
CTA_testis_restricted_gene_ids = cta_testis_restricted_gene_ids
CTA_placental_restricted_gene_names = cta_placental_restricted_gene_names
CTA_placental_restricted_gene_ids = cta_placental_restricted_gene_ids
CTA_relaxed_reproductive_gene_names = cta_relaxed_reproductive_gene_names
CTA_relaxed_reproductive_gene_ids = cta_relaxed_reproductive_gene_ids
CTA_by_axes = cta_by_axes
