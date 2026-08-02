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

"""Gene QC classification for expression matrices.

Answers *which gene-level features are usable* for downstream rescaling: which
rows belong to the one global technical-RNA set: loci whose measured abundance can
be dominated by library preparation (including mitochondrial, rRNA-like, NUMT-like,
nuclear-retained, and structural ncRNA loci). The companion
:func:`oncoref.normalization` helpers consume the same set for every assay.

The classifier is ENSG-first (against oncoref's curated gene-family panels,
stable across symbol renames) then symbol-regex (the self-contained source of
truth). Ported from ``pirlygenes.expression.qc`` with the family lookup rebound to
oncoref's :mod:`oncoref.gene_families` panels — pandas/numpy only, no
pyensembl.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class GeneQcClass:
    """A gene's QC classification: a human-readable ``label`` and a coarse
    ``group`` (the drop-by-default ``group`` drives technical-RNA censoring)."""

    label: str
    group: str


_GENE_NA = {"", "NAN", "NONE", "NULL", "-"}

#: Nuclear-retained, ENE-stabilized lncRNAs that survive degradation
#: disproportionately and creep up as a fraction of bulk TPM (MALAT1, NEAT1).
_POLYA_BIAS_LNCRNA_SYMBOLS = frozenset({"MALAT1", "NEAT1"})

#: The QC groups that constitute "technical RNA" — the drop-by-default / clean-TPM
#: technical compartment (mtDNA, NUMT-like pseudogenes, rRNA-like, polyA-bias lncRNA,
#: and protocol-sensitive structural ncRNA). Membership is global, not selected per assay.
#: Public so a consumer conforming a sample to the clean-TPM space can classify each
#: gene via :func:`classify_gene_qc` and act on ``.group ∈ TECHNICAL_RNA_GROUPS`` —
#: without importing a ``_``-prefixed global.
TECHNICAL_RNA_GROUPS = frozenset(
    {
        "mt_dna",
        "mt_like_pseudogene",
        "rrna_like",
        "polyadenylation_bias_lncrna",
        "protocol_sensitive_structural_rna",
    }
)
#: Back-compat private alias (prefer :data:`TECHNICAL_RNA_GROUPS`).
_TECHNICAL_RNA_GROUPS = TECHNICAL_RNA_GROUPS

#: oncoref gene-family name -> (qc_label, qc_group). The family naming is
#: biological; the QC grouping is the downstream drop-by-default view. (oncoref
#: has no immune-receptor panel — the symbol regex covers IG/TR segments.)
_FAMILY_TO_QC = {
    "mitochondrial": ("mitochondrial transcript", "mt_dna"),
    "numt_pseudogene": ("mitochondrial pseudogene / NUMT-like", "mt_like_pseudogene"),
    "nuclear_retained_lncrna": (
        "nuclear-retained ENE-stabilized lncRNA",
        "polyadenylation_bias_lncrna",
    ),
    "rrna": ("rRNA / rRNA-pseudogene", "rrna_like"),
    "ribosomal_protein": ("ribosomal protein", "ribosomal_protein"),
    "ribosomal_protein_pseudogene": (
        "ribosomal protein pseudogene",
        "ribosomal_protein_pseudogene",
    ),
    "small_noncoding_rna": ("small noncoding RNA", "small_ncrna"),
    "histone": ("histone transcript", "histone"),
    "hemoglobin": ("hemoglobin transcript", "hemoglobin"),
}


@lru_cache(maxsize=1)
def _ensembl_id_to_family() -> dict[str, str]:
    """``{unversioned ENSG -> oncoref gene-family name}`` over the QC families.
    ENSG-first lookup is stable across HGNC symbol renames / version drift."""
    from . import gene_families

    out: dict[str, str] = {}
    for family in _FAMILY_TO_QC:
        if family in gene_families.gene_families():
            for gid in gene_families.gene_family_ids(family):
                out.setdefault(gid, family)
    return out


@lru_cache(maxsize=1)
def _technical_rna_symbols() -> frozenset[str]:
    """Symbols in the single global technical-RNA membership table."""
    from .gene_families import clean_tpm_censored_genes

    rows = clean_tpm_censored_genes()
    rows = rows[rows["category"].astype(str).eq("technical")]
    return frozenset(rows["Symbol"].dropna().astype(str).str.upper())


def _family_to_qc_class(family: str, symbol: str | None = None) -> GeneQcClass:
    """Map a oncoref gene-family name to its QC class, refining the label with a
    symbol-specific sub-classifier when a symbol is given (the QC *group* always
    comes from the family; refinement only affects the human-readable label)."""
    label, group = _FAMILY_TO_QC.get(family, ("protein-coding/other", "other"))
    if symbol:
        refined = _refine_family_label(family, str(symbol).strip().upper())
        if refined is not None:
            label = refined
    return GeneQcClass(label, group)


def _refine_family_label(family: str, upper: str) -> str | None:
    """A more specific human label for ``(family, symbol)`` if available, else None."""
    if family == "mitochondrial":
        if upper in {"MT-RNR1", "MT-RNR2"}:
            return "mitochondrial rRNA"
        if re.fullmatch(r"MT-T[A-Z]\d?", upper):
            return "mitochondrial tRNA"
        return "mitochondrial transcript"
    if family == "rrna":
        if re.fullmatch(r"RNA5SP\d+", upper):
            return "5S rRNA pseudogene"
        if re.fullmatch(r"RNA5-8SP\d+", upper):
            return "5.8S rRNA pseudogene"
        for stem, label in (
            ("RNA18S", "18S rRNA-like"),
            ("RNA28S", "28S rRNA-like"),
            ("RNA45S", "45S pre-rRNA-like"),
            ("RNA5S", "5S rRNA-like"),
        ):
            if upper.startswith(stem):
                return label
    if family == "ribosomal_protein_pseudogene":
        return "ribosomal protein pseudogene"
    if family == "small_noncoding_rna":
        if upper.startswith("SNORD"):
            return "small nucleolar RNA (C/D box)"
        if upper.startswith("SNORA"):
            return "small nucleolar RNA (H/ACA box)"
        if upper.startswith("RNU"):
            return "spliceosomal snRNA"
        if upper.startswith("MIR"):
            return "microRNA"
        if "Y_RNA" in upper or upper.startswith("YR"):
            return "Y RNA"
        if upper.startswith("VTRNA"):
            return "vault RNA"
        if upper.startswith("RN7SK") or upper.startswith("RN7SL"):
            return "signal recognition particle RNA"
    return None


def classify_gene_qc(symbol: str | None = None, *, ensembl_id: str | None = None) -> GeneQcClass:
    """Coarse QC class for a gene by symbol and/or Ensembl ID.

    Lookup order: (1) ENSG against oncoref's curated gene-family panels
    (stable across symbol renames), (2) the symbol regex below (the source of
    truth for the family CSVs). Returns a :class:`GeneQcClass` whose ``group`` is
    one of ``mt_dna``, ``mt_like_pseudogene``, ``rrna_like``, ``ribosomal_protein``,
    ``ribosomal_protein_pseudogene``, ``small_ncrna``,
    ``protocol_sensitive_structural_rna``, ``histone``,
    ``immune_receptor``, ``hemoglobin``, ``polyadenylation_bias_lncrna``, ``other``.
    """
    family = None
    is_global_technical = False
    if ensembl_id:
        from .gene_families import technical_rna_gene_ids
        from .gene_ids import unversioned

        gene_id = unversioned(ensembl_id)
        family = _ensembl_id_to_family().get(gene_id)
        is_global_technical = gene_id in technical_rna_gene_ids()

    raw = str(symbol or "").strip()
    upper = raw.upper()

    if family is not None and not (is_global_technical and family == "small_noncoding_rna"):
        return _family_to_qc_class(family, symbol=upper or None)

    if is_global_technical:
        label = _refine_family_label("small_noncoding_rna", upper) or "structural noncoding RNA"
        return GeneQcClass(label, "protocol_sensitive_structural_rna")

    if upper in _GENE_NA:
        return GeneQcClass("unlabeled feature", "other")

    if upper in _POLYA_BIAS_LNCRNA_SYMBOLS:
        return GeneQcClass("nuclear-retained ENE-stabilized lncRNA", "polyadenylation_bias_lncrna")

    if upper in {"MT-RNR1", "MT-RNR2"}:
        return GeneQcClass("mitochondrial rRNA", "mt_dna")
    if upper.startswith("MT-"):
        return GeneQcClass("mitochondrial transcript", "mt_dna")
    if re.fullmatch(r"MT(RNR[12]|ATP[68]|CO[123]|CYB|ND[1-6]|ND4L)P\d+", upper):
        return GeneQcClass("mitochondrial pseudogene / NUMT-like", "mt_like_pseudogene")

    if re.fullmatch(r"RNA5SP\d+", upper):
        return GeneQcClass("5S rRNA pseudogene", "rrna_like")
    if re.fullmatch(r"RNA5-8SP\d+", upper):
        return GeneQcClass("5.8S rRNA pseudogene", "rrna_like")
    if re.fullmatch(r"RNA(18S|28S|45S|5S)(P\d+|\d+|[_-].*)?", upper):
        label = {
            "RNA18S": "18S rRNA-like",
            "RNA28S": "28S rRNA-like",
            "RNA45S": "45S pre-rRNA-like",
            "RNA5S": "5S rRNA-like",
        }
        prefix = next((p for p in label if upper.startswith(p)), "RNA5S")
        return GeneQcClass(label[prefix], "rrna_like")
    if upper.startswith(("RNR", "MTRNR")):
        return GeneQcClass("rRNA-like", "rrna_like")

    if upper in _technical_rna_symbols():
        label = _refine_family_label("small_noncoding_rna", upper) or "structural noncoding RNA"
        return GeneQcClass(label, "protocol_sensitive_structural_rna")

    if re.fullmatch(r"RP[SL]\d+[A-Z]?(P\d+|P)$", upper):
        return GeneQcClass("ribosomal protein pseudogene", "ribosomal_protein_pseudogene")
    if re.fullmatch(r"RP[SL]\d+[A-Z]?", upper) or upper.startswith("RPLP"):
        return GeneQcClass("ribosomal protein", "ribosomal_protein")

    if upper.startswith(("SNORD", "SNORA", "RNU", "Y_RNA", "MIR")):
        return GeneQcClass("small noncoding RNA", "small_ncrna")

    if upper.startswith(
        ("H1-", "H2AC", "H2BC", "H3C", "H4C", "HIST1H", "HIST2H", "HIST3H", "HIST4H")
    ):
        return GeneQcClass("histone transcript", "histone")

    if re.fullmatch(r"HB(A\d?|B|D|E\d?|G\d?|M|Q\d?|Z|ZP\d?|BP\d?)", upper):
        return GeneQcClass("hemoglobin transcript", "hemoglobin")

    if re.fullmatch(r"(IGH[ADGME]\d*|IG[HKL][CVJ][A-Z0-9-]*|TR[ABDG][CVJ][A-Z0-9-]*)", upper):
        return GeneQcClass("immune receptor segment", "immune_receptor")

    return GeneQcClass("protein-coding/other", "other")


def is_rescue_feature(symbol: str | None = None, *, ensembl_id: str | None = None) -> bool:
    """True when a feature is technical RNA (removed by mtDNA/rRNA censoring) — i.e.
    its QC group is in the drop-by-default technical-RNA set."""
    return classify_gene_qc(symbol, ensembl_id=ensembl_id).group in TECHNICAL_RNA_GROUPS


__all__ = [
    "GeneQcClass",
    "classify_gene_qc",
    "is_rescue_feature",
]
