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

"""Gene-family reference lists for expression normalization.

The curated families behind oncoref's normalization, plus the single global
clean-TPM censored-gene table. That table includes mtDNA, NUMT pseudogenes, rRNA,
nuclear-retained lncRNAs, and protocol-sensitive structural ncRNAs with their
clean-PolyA reference TPMs. This is the read surface; the clean-TPM engine that
consumes it lives in :mod:`oncoref.normalization`.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .load_dataset import get_data

#: family name -> bundled dataset.
_FAMILIES = {
    "histone": "histone-genes",
    "ribosomal_protein": "ribosomal-protein-genes",
    "ribosomal_protein_pseudogene": "ribosomal-protein-pseudogenes",
    "mitochondrial": "mitochondrial-genes",
    "rrna": "rrna-and-pseudogenes",
    "numt_pseudogene": "numt-pseudogenes",
    "small_noncoding_rna": "small-noncoding-rnas",
    "nuclear_retained_lncrna": "nuclear-retained-lncrnas",
    "hemoglobin": "hemoglobin-genes",
}

#: Historical curated families contained in the global technical-RNA list. The complete
#: assay-independent membership is :func:`technical_rna_gene_ids`; structural ncRNAs are
#: derived from canonical biotypes rather than pretending they are all one gene family.
TECHNICAL_RNA_FAMILIES = ("mitochondrial", "numt_pseudogene", "rrna", "nuclear_retained_lncrna")
#: Back-compat private alias (prefer :data:`TECHNICAL_RNA_FAMILIES`).
_TECHNICAL_RNA_FAMILIES = TECHNICAL_RNA_FAMILIES

#: Canonical Ensembl biotypes for structural, generally non-polyadenylated RNAs whose
#: measured abundance can be artifactually over-represented by 10x or more under a
#: different library preparation. These genes belong in
#: clean TPM's other-technical compartment so ribo-depleted and polyA-selected libraries
#: can share the biological 75% scale. miRNA and lncRNA are deliberately absent: this is
#: a structural-RNA protocol correction, not blanket small-noncoding-RNA censorship.
CLEAN_TPM_PROTOCOL_SENSITIVE_BIOTYPES = frozenset(
    {"misc_RNA", "ribozyme", "scaRNA", "snoRNA", "snRNA", "sRNA", "vault_RNA"}
)

#: Provenance for the fixed censored-gene composition used by clean TPM.
CLEAN_TPM_CENSORED_REFERENCE_PROFILE_VERSION = "treehouse_polya_25_01_median_v1"
CLEAN_TPM_CENSORED_REFERENCE_PROFILE_SOURCE = "Treehouse Tumor Compendium 25.01 PolyA median TPM"

#: Default HPA RNA floor for empirical housekeeping-panel candidates.
#: The panel is meant to provide a robust denominator, so low-abundance
#: qPCR-style references are excluded even if they are broad.
HPA_HOUSEKEEPING_MIN_NTPM: float = 100.0
#: Maximum tissue coefficient of variation for HPA-derived candidates.
HPA_HOUSEKEEPING_MAX_CV: float = 0.5
#: Preferred maximum tissue max/min range for the primary panel.
HPA_HOUSEKEEPING_MAX_MIN_RATIO: float = 6.5
#: Deliberate high-abundance/literature-supported range exceptions.
HPA_HOUSEKEEPING_RANGE_EXCEPTION_IDS = frozenset({"ENSG00000156508"})  # EEF1A1
#: Biologically plausible but held out of the first-pass denominator.
HPA_HOUSEKEEPING_HOLDOUT_IDS = frozenset(
    {
        "ENSG00000196262",  # PPIA: passes numerically, but context/paralog concerns.
        "ENSG00000080824",  # HSP90AA1: redundant with HSP90AB1.
    }
)


def gene_families() -> tuple[str, ...]:
    """The available gene-family names."""
    return tuple(_FAMILIES)


def gene_family(name: str) -> pd.DataFrame:
    """The table for one gene family (``Symbol``, ``Ensembl_Gene_ID``, …). Copy."""
    try:
        dataset = _FAMILIES[name]
    except KeyError:
        raise ValueError(f"unknown gene family {name!r}; one of {sorted(_FAMILIES)}") from None
    return get_data(dataset).copy()


def _unversioned_ids(df: pd.DataFrame) -> frozenset[str]:
    ids = df["Ensembl_Gene_ID"].dropna().astype(str).str.split(".").str[0]
    return frozenset(ids)


@lru_cache(maxsize=len(_FAMILIES))
def gene_family_ids(name: str) -> frozenset[str]:
    """Unversioned Ensembl IDs of one gene family."""
    if name not in _FAMILIES:
        raise ValueError(f"unknown gene family {name!r}; one of {sorted(_FAMILIES)}")
    return _unversioned_ids(get_data(_FAMILIES[name], copy=False))


@lru_cache(maxsize=1)
def technical_rna_gene_ids() -> frozenset[str]:
    """The one global technical-RNA gene set used for every assay.

    This is exactly clean TPM's ``technical`` category: the historical mtDNA / NUMT /
    rRNA / nuclear-retained families plus protocol-sensitive structural ncRNAs.
    """
    return clean_tpm_other_technical_gene_ids()


@lru_cache(maxsize=1)
def clean_tpm_protocol_sensitive_gene_ids() -> frozenset[str]:
    """Canonical IDs of protocol-sensitive structural noncoding RNAs.

    Membership is derived from the bundled canonical gene-space biotypes listed in
    :data:`CLEAN_TPM_PROTOCOL_SENSITIVE_BIOTYPES`. The same IDs are materialized in
    ``clean-tpm-censored-genes.csv`` so the runtime compartment contract remains one
    explicit, inspectable table.
    """
    genes = get_data("canonical-gene-space", copy=False)
    selected = genes["biotype"].astype(str).isin(CLEAN_TPM_PROTOCOL_SENSITIVE_BIOTYPES)
    return frozenset(genes.loc[selected, "ensembl_gene_id"].astype(str).str.split(".").str[0])


def housekeeping_genes() -> pd.DataFrame:
    """The legacy qPCR/reference housekeeping panel."""
    return get_data("housekeeping-genes").copy()


@lru_cache(maxsize=1)
def housekeeping_gene_ids() -> frozenset[str]:
    """Unversioned Ensembl IDs of the housekeeping panel."""
    return _unversioned_ids(get_data("housekeeping-genes", copy=False))


def legacy_qpcr_housekeeping_genes() -> pd.DataFrame:
    """Classic qPCR/reference-gene housekeeping panel.

    This is the historical :func:`housekeeping_genes` table. It intentionally remains
    available, but clean-TPM biological denominators should use
    :func:`clean_tpm_biological_housekeeping_genes` instead because this legacy panel
    includes ribosomal-protein genes that live in clean TPM's 16% ribosomal compartment.
    """
    return housekeeping_genes()


def legacy_qpcr_housekeeping_gene_ids() -> frozenset[str]:
    """Unversioned Ensembl IDs of the classic qPCR/reference-gene panel."""
    return housekeeping_gene_ids()


def _primary_panel_mask(df: pd.DataFrame) -> pd.Series:
    return df["primary_panel"].astype(str).str.lower().isin({"true", "1", "yes"})


def clean_tpm_biological_housekeeping_genes(*, primary_only: bool = True) -> pd.DataFrame:
    """HPA-stable biological housekeeping genes for clean-TPM denominators.

    The bundled table contains all 47 strict HPA whole-tissue candidates that are
    protein-coding, clean-TPM biological genes with ``min_tpm >= 100`` across 50 HPA
    tissues and low tissue coefficient of variation. By default this returns the
    curated 30-gene primary denominator panel. Pass ``primary_only=False`` for the
    full candidate/audit table including HPA provenance and exclusion notes.
    """
    df = get_data("clean-tpm-biological-housekeeping-genes")
    if primary_only:
        df = df[_primary_panel_mask(df)]
    return df.reset_index(drop=True)


@lru_cache(maxsize=2)
def clean_tpm_biological_housekeeping_gene_ids(*, primary_only: bool = True) -> frozenset[str]:
    """Unversioned Ensembl IDs for the clean-TPM biological housekeeping panel.

    Defaults to the 30-gene primary denominator panel. Pass ``primary_only=False``
    to include every strict HPA-stable candidate in the shipped audit table.
    """
    return _unversioned_ids(clean_tpm_biological_housekeeping_genes(primary_only=primary_only))


@lru_cache(maxsize=2)
def clean_tpm_censored_gene_ids(*, include_ribosomal_proteins: bool = True) -> frozenset[str]:
    """Unversioned Ensembl IDs censored by the clean-TPM transform — technical RNA
    plus, by default, ribosomal-protein genes. Pass
    ``include_ribosomal_proteins=False`` for the strict technical-only set.

    **Curated, biology-defined membership — NOT data-derived.** Membership is the technical-
    RNA families + ribosomal proteins: genes that are non-mRNA or library-prep artifacts by
    their molecular nature. Never expand this set from expression variance or abundance —
    cancer-testis antigens are high-variance *by definition* (silent in most samples, high in
    a few, which is exactly what makes them targets), so a variance-based rule would censor the
    very antigens oncoref exists to find. Data (TCGA LUAD/SKCM) is used only to *calibrate*
    the clean-TPM compartment fractions and to *validate completeness* of this list (how the
    missing rRNA genes, incl. 28S, were found)."""
    df = _clean_tpm_censored_genes()
    if not include_ribosomal_proteins:
        df = df[df["category"].astype(str) == "technical"]
    return _unversioned_ids(df)


@lru_cache(maxsize=1)
def clean_tpm_ribosomal_gene_ids() -> frozenset[str]:
    """Unversioned Ensembl IDs assigned to clean TPM's 16% ribosomal compartment.

    This is defined by ``clean-tpm-censored-genes.csv`` where
    ``category == "ribosomal_protein"``, not by the broader ribosomal-protein family
    table. That distinction is CTA-safe: broad ribosomal-family members that are absent
    from the censored table, such as ``RPL10L`` / ``ENSG00000165496``, remain biological.
    """
    df = _clean_tpm_censored_genes()
    df = df[df["category"].astype(str) == "ribosomal_protein"]
    return _unversioned_ids(df)


@lru_cache(maxsize=1)
def clean_tpm_other_technical_gene_ids() -> frozenset[str]:
    """Unversioned Ensembl IDs assigned to clean TPM's 9% other-technical compartment.

    This is defined by ``clean-tpm-censored-genes.csv`` where
    ``category == "technical"``.
    """
    df = _clean_tpm_censored_genes()
    df = df[df["category"].astype(str) == "technical"]
    return _unversioned_ids(df)


@lru_cache(maxsize=1)
def _clean_tpm_censored_genes() -> pd.DataFrame:
    columns = {"Ensembl_Gene_ID", "Symbol", "category", "reference_tpm"}
    df = get_data("clean-tpm-censored-genes", copy=False)
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"clean-tpm-censored-genes is missing columns: {missing}")
    return df


def clean_tpm_censored_genes() -> pd.DataFrame:
    """The single global censored-gene membership and PolyA reference profile.

    ``category`` assigns the 16% ribosomal or 9% other-technical compartment;
    ``reference_tpm`` fixes composition within that compartment to the Treehouse
    25.01 PolyA median. The table is assay-independent and returned as a copy.
    """
    return _clean_tpm_censored_genes().copy()


@lru_cache(maxsize=1)
def clean_tpm_censored_reference_tpm() -> dict[str, float]:
    """``{unversioned Ensembl ID: PolyA reference TPM}`` for all censored genes."""
    df = _clean_tpm_censored_genes()
    return {
        str(gene_id).split(".")[0]: float(value)
        for gene_id, value in zip(df["Ensembl_Gene_ID"], df["reference_tpm"])
    }


@lru_cache(maxsize=1)
def censored_gene_reference_tpm() -> dict[str, float]:
    """Legacy ``{Symbol: reference_tpm}`` view of the global censored profile."""
    df = _clean_tpm_censored_genes().drop_duplicates("Symbol")
    return {str(s): float(v) for s, v in zip(df["Symbol"], df["reference_tpm"])}


def _unversioned_series(values: pd.Series) -> pd.Series:
    return values.dropna().astype(str).str.split(".").str[0]


def _protein_coding_gene_ids(gene_space: pd.DataFrame | None = None) -> frozenset[str]:
    """Protein-coding Ensembl gene IDs from the canonical gene-space table."""
    if gene_space is None:
        gene_space = get_data("canonical-gene-space", copy=False)
    gid_col = "ensembl_gene_id" if "ensembl_gene_id" in gene_space.columns else "Ensembl_Gene_ID"
    if gid_col not in gene_space.columns or "biotype" not in gene_space.columns:
        raise ValueError("gene_space must contain an Ensembl ID column and 'biotype'")
    ids = _unversioned_series(
        gene_space.loc[gene_space["biotype"].astype(str).eq("protein_coding"), gid_col]
    )
    return frozenset(ids)


def hpa_housekeeping_candidates(
    hpa_rna: pd.DataFrame | None = None,
    *,
    gene_space: pd.DataFrame | None = None,
    min_ntpm: float = HPA_HOUSEKEEPING_MIN_NTPM,
    max_cv: float = HPA_HOUSEKEEPING_MAX_CV,
    protein_coding_only: bool = True,
    biological_only: bool = True,
) -> pd.DataFrame:
    """Empirically derive broad housekeeping-like candidates from HPA tissue RNA.

    The input is the long HPA RNA consensus table with columns ``Gene``, ``Gene name``,
    ``Tissue``, and ``nTPM``. By default the function loads oncoref's pinned HPA
    consensus data and filters to protein-coding, clean-TPM-biological genes with
    complete tissue coverage, ``min(nTPM) >= 100``, and tissue ``CV < 0.5``.

    This is intentionally a candidate scorer, not a replacement for biological review
    or a drop-in replacement for the active housekeeping denominator. Changing the
    panel changes the scale of every HK-normalized expression value, so downstream
    thresholds must be recalibrated and compared against clean TPM, log1p(clean TPM),
    and percentile-rank clean TPM before this is promoted to the bundled panel.
    """
    if hpa_rna is None:
        from .hpa import hpa_rna_consensus

        hpa_rna = hpa_rna_consensus()

    required = {"Gene", "Gene name", "Tissue", "nTPM"}
    missing = required - set(hpa_rna.columns)
    if missing:
        raise ValueError(f"hpa_rna missing required columns: {sorted(missing)}")

    expr = hpa_rna.loc[:, ["Gene", "Gene name", "Tissue", "nTPM"]].copy()
    expr["Ensembl_Gene_ID"] = _unversioned_series(expr["Gene"])
    expr["Symbol"] = expr["Gene name"].astype(str).str.strip()
    expr["Tissue"] = expr["Tissue"].astype(str).str.strip()
    expr["nTPM"] = pd.to_numeric(expr["nTPM"], errors="coerce")
    expr = expr.dropna(subset=["Ensembl_Gene_ID", "Symbol", "Tissue"])
    n_expected_tissues = expr["Tissue"].nunique()
    expr = expr.dropna(subset=["nTPM"])

    if protein_coding_only:
        expr = expr[expr["Ensembl_Gene_ID"].isin(_protein_coding_gene_ids(gene_space))]
    if biological_only:
        expr = expr[~expr["Ensembl_Gene_ID"].isin(clean_tpm_censored_gene_ids())]

    # Average duplicate gene/tissue rows defensively, then compute population CV
    # directly from the long table so callers do not need the old wide HPA matrix.
    expr = expr.groupby(["Ensembl_Gene_ID", "Symbol", "Tissue"], as_index=False)["nTPM"].mean()
    expr["_ntpm_sq"] = expr["nTPM"] ** 2
    stats = expr.groupby(["Ensembl_Gene_ID", "Symbol"], as_index=False).agg(
        min_ntpm=("nTPM", "min"),
        mean_ntpm=("nTPM", "mean"),
        max_ntpm=("nTPM", "max"),
        mean_ntpm_sq=("_ntpm_sq", "mean"),
        n_tissues=("Tissue", "nunique"),
    )
    stats["n_tissues_expected"] = n_expected_tissues
    variance = (stats["mean_ntpm_sq"] - stats["mean_ntpm"] ** 2).clip(lower=0.0)
    denom = stats["mean_ntpm"].where(stats["mean_ntpm"] > 0)
    stats["cv"] = (variance**0.5) / denom
    stats["max_min_ratio"] = stats["max_ntpm"] / stats["min_ntpm"].where(stats["min_ntpm"] > 0)
    stats = stats.drop(columns=["mean_ntpm_sq"])
    complete = stats["n_tissues"] == stats["n_tissues_expected"]
    stats = stats[complete & (stats["min_ntpm"] >= min_ntpm) & (stats["cv"] < max_cv)]
    stats = stats.sort_values(["cv", "max_min_ratio", "Symbol"], kind="stable").reset_index(
        drop=True
    )
    stats.insert(0, "rank", range(1, len(stats) + 1))
    return stats


def recommended_hpa_housekeeping_panel(
    hpa_rna: pd.DataFrame | None = None,
    *,
    gene_space: pd.DataFrame | None = None,
    target_size: int | None = 30,
    min_ntpm: float = HPA_HOUSEKEEPING_MIN_NTPM,
    max_cv: float = HPA_HOUSEKEEPING_MAX_CV,
    max_min_ratio: float = HPA_HOUSEKEEPING_MAX_MIN_RATIO,
    range_exception_ids: frozenset[str] = HPA_HOUSEKEEPING_RANGE_EXCEPTION_IDS,
    holdout_ids: frozenset[str] = HPA_HOUSEKEEPING_HOLDOUT_IDS,
) -> pd.DataFrame:
    """Recommended HPA-derived primary housekeeping denominator panel.

    The policy is reproducible and intentionally conservative:

    - start with :func:`hpa_housekeeping_candidates`;
    - prefer a compressed cross-tissue range (``max/min <= 6.5``);
    - allow explicit high-abundance/literature-supported range exceptions;
    - hold out candidates with known biological/paralog redundancy concerns.

    The returned table is still an empirical recommendation. Source-library maintainers
    should review the exclusion/exception constants and recalibrate every consumer that
    interprets HK-normalized magnitudes before promoting it to the bundled
    ``housekeeping-genes.csv`` table. Many analyses may be better served by clean TPM,
    log1p(clean TPM), or percentile-rank clean TPM instead of HK normalization.
    """
    candidates = hpa_housekeeping_candidates(
        hpa_rna,
        gene_space=gene_space,
        min_ntpm=min_ntpm,
        max_cv=max_cv,
        protein_coding_only=True,
        biological_only=True,
    )
    exceptions = {str(g).split(".")[0] for g in range_exception_ids}
    holdouts = {str(g).split(".")[0] for g in holdout_ids}
    gids = candidates["Ensembl_Gene_ID"].astype(str)
    keep = ~gids.isin(holdouts) & (
        candidates["max_min_ratio"].le(max_min_ratio) | gids.isin(exceptions)
    )
    out = candidates.loc[keep].copy()
    out["selection_reason"] = "strict_hpa_range"
    out.loc[out["Ensembl_Gene_ID"].isin(exceptions), "selection_reason"] = (
        "high_abundance_literature_exception"
    )

    if target_size is not None and len(out) > target_size:
        protected = out[out["Ensembl_Gene_ID"].isin(exceptions)]
        regular = out[~out["Ensembl_Gene_ID"].isin(exceptions)].head(
            max(target_size - len(protected), 0)
        )
        out = pd.concat([regular, protected], ignore_index=True)
        out = out.sort_values(["cv", "max_min_ratio", "Symbol"], kind="stable")

    out = out.reset_index(drop=True)
    out.insert(0, "panel_rank", range(1, len(out) + 1))
    return out
