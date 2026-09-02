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

"""Characteristic gene fusions / oncogenic translocations per cancer type.

The curated ``cancer-fusions.csv`` reference: one row per concrete 5'/3' gene
pairing, with the partner protein families annotated (FET, ETS, PAX, FOX, MiT/TFE,
RTK, Ig locus, …), an ``is_defining`` flag for the characteristic lesion of the
entity, and the stronger ``pathognomonic`` flag for pairs that map to a single
entity. This module is the read + query surface: per-type lookup, reverse lookup
(which types carry a fusion / partner / partner family), and partner sets.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .cancer_types import cancer_type_descendants, resolve_cancer_type
from .gene_ids import canonical_gene_id
from .legacy import warn_legacy_dataset_access
from .load_dataset import _register_derived_cache, get_data

FUSION_PARTNER_KINDS = frozenset({"gene", "immunoglobulin_locus", "tcr_locus", "none"})
IMMUNOGLOBULIN_FUSION_LOCI = frozenset({"IGH", "IGK", "IGL"})
TCR_FUSION_LOCI = frozenset({"TRA", "TRB", "TRD", "TRG"})


def _clean(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _validate_fusion_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cancer_code",
        "fusion_family",
        "gene_5prime",
        "gene_5prime_ensembl_id",
        "gene_5prime_kind",
        "gene_3prime",
        "gene_3prime_ensembl_id",
        "gene_3prime_kind",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"cancer-fusions is missing required columns: {missing}")

    duplicate_key = ["cancer_code", "gene_5prime", "gene_3prime"]
    duplicate_rows = frame.duplicated(duplicate_key, keep=False)
    if duplicate_rows.any():
        keys = frame.loc[duplicate_rows, duplicate_key].fillna("").to_dict("records")
        raise ValueError(f"cancer-fusions contains duplicate logical keys: {keys}")

    for index, row in frame.iterrows():
        for side in ("5prime", "3prime"):
            symbol = _clean(row[f"gene_{side}"]).upper()
            ensembl_id = _clean(row[f"gene_{side}_ensembl_id"])
            kind = _clean(row[f"gene_{side}_kind"])
            if kind not in FUSION_PARTNER_KINDS:
                raise ValueError(
                    f"cancer-fusions row {index} has invalid gene_{side}_kind {kind!r}"
                )
            if kind == "none":
                if symbol or ensembl_id:
                    raise ValueError(
                        f"cancer-fusions row {index} marks a named partner as kind='none'"
                    )
                continue
            if kind == "immunoglobulin_locus":
                if symbol not in IMMUNOGLOBULIN_FUSION_LOCI or ensembl_id:
                    raise ValueError(
                        f"cancer-fusions row {index} has an invalid immunoglobulin locus"
                    )
                continue
            if kind == "tcr_locus":
                if symbol not in TCR_FUSION_LOCI or ensembl_id:
                    raise ValueError(f"cancer-fusions row {index} has an invalid TCR locus")
                continue
            expected = canonical_gene_id(symbol)
            if expected is None:
                raise ValueError(
                    f"cancer-fusions row {index} has unresolved gene partner {symbol!r}"
                )
            if ensembl_id != expected or canonical_gene_id(ensembl_id) != expected:
                raise ValueError(
                    f"cancer-fusions row {index} has a symbol/Ensembl mismatch for "
                    f"{symbol}: expected {expected}, observed {ensembl_id or '<missing>'}"
                )
    return frame


@lru_cache(maxsize=1)
def _cancer_fusions_frame() -> pd.DataFrame:
    return _validate_fusion_frame(get_data("cancer-fusions", copy=False))


_register_derived_cache(_cancer_fusions_frame.cache_clear)


def cancer_fusions_df():
    """The full curated fusion table (defensive copy).

    Columns: ``cancer_code``, ``fusion_family``, ``gene_5prime``,
    ``gene_5prime_ensembl_id``, ``gene_5prime_kind``, ``gene_5prime_family``,
    ``gene_3prime``, ``gene_3prime_ensembl_id``, ``gene_3prime_kind``,
    ``gene_3prime_family``, ``frequency``, ``is_defining``, ``pathognomonic``,
    ``rnaseq_detectable``, ``mechanism``, ``confidence``, ``pmid``, ``notes``.
    Partner kinds distinguish ordinary genes, immunoglobulin loci, TCR loci, and
    absent partners. Fusion-negative entities carry a single
    ``fusion_family="(none)"`` row naming the real driver.
    """
    return _cancer_fusions_frame().copy()


def cancer_fusion_citation_audit():
    """Reviewed PubMed evidence for every cited row in :func:`cancer_fusions_df`.

    The table records the PubMed title and review date, plus separate flags for
    support of the alteration and its disease context. It is kept separate from
    the fusion table so citation review metadata does not complicate fusion
    queries.
    """
    return get_data("cancer-fusion-citation-audit").copy()


def _truthy(series):
    # is_defining / pathognomonic load as bool, but compare via lowercased string
    # so the filter is robust whether the column is bool or str.
    return series.astype(str).str.lower() == "true"


def _upper(series):
    """Uppercased view of a gene/family column with NaN preserved as non-matching.

    The fusion-negative ``"(none)"`` rows carry NaN gene names; ``.str.upper()``
    keeps those NaN (so ``NaN == "X"`` is False) instead of ``.astype(str)``
    turning them into the literal string ``"NAN"`` that a query could match.
    """
    return series.str.upper()


def cancer_fusions(
    cancer_type=None, *, defining_only=False, pathognomonic_only=False, include_subtypes=False
):
    """Fusion rows for one cancer type (alias-resolved), or the whole table when
    ``cancer_type`` is None.

    ``defining_only`` / ``pathognomonic_only`` filter to the characteristic /
    diagnostic rows. ``include_subtypes`` also pulls the fusions of every code in
    the cancer type's subtree (via :func:`cancer_type_descendants`) — e.g. the
    union over an ``SARC_RMS`` parent once its subtypes are parented under it.
    """
    df = cancer_fusions_df()
    if cancer_type is not None:
        code = resolve_cancer_type(cancer_type, strict=False) or cancer_type
        codes = {code}
        if include_subtypes:
            codes |= set(cancer_type_descendants(code))
        df = df[df["cancer_code"].astype(str).isin(codes)]
    if defining_only:
        df = df[_truthy(df["is_defining"])]
    if pathognomonic_only:
        df = df[_truthy(df["pathognomonic"])]
    return df.reset_index(drop=True)


def fusion_partners(gene, *, side=None):
    """The set of fusion partners of ``gene`` in the table.

    ``side=None`` returns partners on either end; ``side="5prime"`` the observed
    3' partners when ``gene`` is 5'; ``side="3prime"`` the observed 5' partners.
    """
    if side not in (None, "5prime", "3prime"):
        raise ValueError("side must be None, '5prime', or '3prime'")
    g = str(gene).strip().upper()
    df = cancer_fusions_df()
    out = set()
    if side in (None, "5prime"):
        out |= set(df.loc[_upper(df["gene_5prime"]) == g, "gene_3prime"])
    if side in (None, "3prime"):
        out |= set(df.loc[_upper(df["gene_3prime"]) == g, "gene_5prime"])
    return {p for p in out if isinstance(p, str) and p.strip()}


def cancer_types_with_fusion(
    fusion=None, *, partner=None, partner_family=None, defining_only=False, as_rows=False
):
    """Reverse fusion lookup — the inverse of :func:`cancer_fusions`.

    Pass exactly one of:
    - ``fusion="EWSR1-FLI1"`` — a directional ``5'-3'`` string (``::`` also accepted);
    - ``partner="EWSR1"`` — a partner gene on either end;
    - ``partner_family="FET"`` — a partner-family tag.

    ``defining_only`` restricts to is_defining rows. Returns sorted cancer codes,
    or the matching rows when ``as_rows=True``.
    """
    given = [x for x in (fusion, partner, partner_family) if x is not None]
    if len(given) != 1:
        raise ValueError("pass exactly one of fusion=, partner=, or partner_family=")
    df = cancer_fusions(defining_only=defining_only)
    g5 = _upper(df["gene_5prime"])
    g3 = _upper(df["gene_3prime"])
    if fusion is not None:
        parts = str(fusion).upper().replace("::", "-").split("-")
        if len(parts) != 2:
            raise ValueError(f"fusion must look like '5GENE-3GENE'; got {fusion!r}")
        a, b = (p.strip() for p in parts)
        mask = (g5 == a) & (g3 == b)
    elif partner is not None:
        p = str(partner).strip().upper()
        mask = (g5 == p) | (g3 == p)
    else:
        fam = str(partner_family).strip().upper()
        f5 = _upper(df["gene_5prime_family"])
        f3 = _upper(df["gene_3prime_family"])
        mask = (f5 == fam) | (f3 == fam)
    mask = mask.fillna(False)
    hits = df[mask]
    if as_rows:
        return hits.reset_index(drop=True)
    return sorted({str(c) for c in hits["cancer_code"] if str(c).strip()})


def protein_family(gene):
    """Protein/gene family of a fusion partner (EWSR1→FET, FLI1→ETS, PAX3→PAX,
    FOXO1→FOX, ALK→RTK), or ``None`` if the gene has no family annotation."""
    g = str(gene).strip().upper()
    df = cancer_fusions_df()
    for col, fam in (("gene_5prime", "gene_5prime_family"), ("gene_3prime", "gene_3prime_family")):
        hit = df.loc[_upper(df[col]) == g, fam]
        for v in hit:
            if isinstance(v, str) and v.strip():
                return v
    return None


# ---------- fusion rule / surrogate / effect tables (R-onto) ----------


def rare_cancer_fusion_rules_df():
    """Frozen compatibility snapshot of direct rare-cancer fusion rules.

    New per-sample interpretation rules belong to trufflepig. Columns include
    ``rule_id``, ``cancer_code``, ``gene_a``, ``gene_b``, ``matching``, and
    ``confidence``. Defensive copy.
    """
    warn_legacy_dataset_access("rare-cancer-fusion-rules", stacklevel=2)
    return get_data("rare-cancer-fusion-rules").copy()


def fusion_surrogate_expression_df():
    """Frozen compatibility snapshot of fusion-expression surrogate rules.

    New per-sample surrogate interpretation belongs to trufflepig. Columns
    include ``fusion_class``, ``surrogate_gene``, ``surrogate_role``, and
    ``cancer_code``.
    """
    warn_legacy_dataset_access("fusion-surrogate-expression", stacklevel=2)
    return get_data("fusion-surrogate-expression").copy()


def fusion_expression_effect_rules_df():
    """Frozen compatibility snapshot of fusion-expression effect rules.

    New per-sample effect interpretation belongs to trufflepig. Columns include
    ``gene_a``, ``gene_b``, ``anchor_genes``, and ``expected_up_genes``.
    Defensive copy.
    """
    warn_legacy_dataset_access("fusion-expression-effects", stacklevel=2)
    return get_data("fusion-expression-effects").copy()


def fusion_surrogate_genes_for_cancer(cancer_type):
    """Surrogate gene symbols whose expression flags a fusion in ``cancer_type``."""
    from .cancer_types import resolve_cancer_type

    code = resolve_cancer_type(cancer_type)
    df = fusion_surrogate_expression_df()
    return sorted(
        {
            str(s)
            for s, c in zip(df["surrogate_gene"], df["cancer_code"])
            if str(c) == code and isinstance(s, str) and s.strip()
        }
    )
