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

"""Per-cancer-type gene biology: drivers, key (biomarker/target) genes, role-
stratified type genes, viral antigens, and a few narrative/rule tables.

The curated ontology metadata that hangs off the cancer-type registry. All code
arguments are alias-resolved via :func:`oncoref.resolve_cancer_type`.
"""

from __future__ import annotations

import pandas as pd

from .cancer_types import resolve_cancer_type
from .legacy import warn_legacy_dataset_access
from .load_dataset import get_data


def _split(value, sep=";") -> list[str]:
    return [x.strip() for x in str(value).split(sep) if x.strip() and x.strip().lower() != "nan"]


# ---------- driver genes / variants ----------


def cancer_driver_genes_df() -> pd.DataFrame:
    """Frozen driver-gene compatibility snapshot; defensive copy.

    Use :func:`oncoref.drivers.driver_gene_evidence_df` for the source-anchored
    replacement. Legacy columns include ``Symbol``, ``Cancer``, ``Function``,
    and ``Ensembl_Gene_ID``.
    """
    warn_legacy_dataset_access("cancer-driver-genes", stacklevel=2)
    return get_data("cancer-driver-genes").copy()


def cancer_driver_variants_df() -> pd.DataFrame:
    """Frozen driver-variant compatibility snapshot; defensive copy.

    Use :func:`oncoref.drivers.driver_variant_evidence_df` for the
    source-anchored replacement. Legacy columns include ``Symbol``, ``Mutation``,
    ``Ensembl_Transcript_ID``, and ``Ensembl_Gene_ID``.
    """
    warn_legacy_dataset_access("cancer-driver-variants", stacklevel=2)
    return get_data("cancer-driver-variants").copy()


# ---------- key genes: biomarkers + therapy targets ----------


def cancer_key_genes_df() -> pd.DataFrame:
    """Frozen compatibility snapshot of per-type biomarker/target rows.

    New purpose-specific panels belong to pirlygenes; source-anchored clinical
    benefit/toxicity facts belong to oncoref's therapy-evidence API. Defensive
    copy.
    """
    warn_legacy_dataset_access("cancer-key-genes", stacklevel=2)
    return get_data("cancer-key-genes").copy()


def cancer_key_gene_citation_audit() -> pd.DataFrame:
    """Reviewed PubMed evidence for each citation in :func:`cancer_key_genes_df`.

    Each row records one cited PMID and confirms that it supports the named
    gene or agent, disease context, role, and therapeutic phase. Defensive copy.
    """
    return get_data("cancer-key-gene-citation-audit").copy()


def _key_genes_for(cancer_type, *, subtype=None) -> pd.DataFrame:
    df = cancer_key_genes_df()
    df = df[df["cancer_code"].astype(str) == resolve_cancer_type(cancer_type)]
    if subtype is not None:
        df = df[df["subtype"].astype(str) == str(subtype)]
    return df


def cancer_biomarker_genes(cancer_type, *, subtype=None) -> list[str]:
    """Biomarker gene symbols for a cancer type (ordered, de-duplicated)."""
    df = _key_genes_for(cancer_type, subtype=subtype)
    syms = df[df["role"].astype(str) == "biomarker"]["symbol"].astype(str)
    return list(dict.fromkeys(syms))


def cancer_therapy_targets(cancer_type, *, subtype=None) -> pd.DataFrame:
    """Therapy-target rows for a cancer type (agent / phase / indication)."""
    df = _key_genes_for(cancer_type, subtype=subtype)
    return df[df["role"].astype(str) == "target"].reset_index(drop=True)


# ---------- role-stratified type genes ----------


def cancer_type_genes_df() -> pd.DataFrame:
    """Frozen role-stratified cancer-gene compatibility snapshot.

    New purpose-specific lineage and marker panels belong to pirlygenes. Legacy
    columns include ``Symbol``, ``Ensembl_Gene_ID``, ``Cancer_Type``, and
    ``Role``. Defensive copy.
    """
    warn_legacy_dataset_access("cancer-type-genes", stacklevel=2)
    return get_data("cancer-type-genes").copy()


def cancer_type_gene_sets(cancer_type) -> dict[str, dict[str, str]]:
    """``{role: {ensembl_id: symbol}}`` for one cancer type (empty if none curated)."""
    code = resolve_cancer_type(cancer_type)
    df = cancer_type_genes_df()
    df = df[df["Cancer_Type"].astype(str) == code]
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        out.setdefault(str(row["Role"]), {})[str(row["Ensembl_Gene_ID"])] = str(row["Symbol"])
    return out


# ---------- viral antigens ----------


def cancer_viral_antigens_df() -> pd.DataFrame:
    """Frozen per-oncovirus target-selection compatibility snapshot.

    New oncovirus target-selection panels belong to pirlygenes. Legacy columns
    include ``virus``, ``targetable_antigens``, and ``associated_cohorts``.
    Defensive copy.
    """
    warn_legacy_dataset_access("cancer-viral-antigens", stacklevel=2)
    return get_data("cancer-viral-antigens").copy()


def cancer_viral_antigens(virus: str | None = None):
    """Targetable viral antigens. With ``virus`` (case-insensitive), the list for
    that virus (``[]`` if unknown); otherwise a ``{virus: [antigen, …]}`` map."""
    df = cancer_viral_antigens_df()
    if virus is not None:
        hit = df[df["virus"].astype(str).str.lower() == str(virus).strip().lower()]
        return _split(hit.iloc[0]["targetable_antigens"]) if not hit.empty else []
    return {str(r.virus): _split(r.targetable_antigens) for r in df.itertuples()}


def viral_antigens_for_cancer(cancer_type) -> list[tuple[str, list[str]]]:
    """``[(virus, [antigen, …]), …]`` for a cancer type — the reverse lookup over
    ``associated_cohorts``. Empty for a non-virally-driven entity."""
    code = resolve_cancer_type(cancer_type)
    out = []
    for r in cancer_viral_antigens_df().itertuples():
        if code in _split(r.associated_cohorts):
            out.append((str(r.virus), _split(r.targetable_antigens)))
    return out


# ---------- narrative / rule tables ----------


def narrative_gene_sets_df() -> pd.DataFrame:
    """Frozen named-gene-set compatibility snapshot.

    New purpose-specific named gene sets belong to pirlygenes. Legacy columns
    include ``set_name``, ``members``, and ``notes``. Defensive copy.
    """
    warn_legacy_dataset_access("narrative-gene-sets", stacklevel=2)
    return get_data("narrative-gene-sets").copy()


def narrative_gene_set(set_name: str) -> list[str]:
    """Member gene symbols of a named narrative set (``[]`` if unknown)."""
    df = narrative_gene_sets_df()
    hit = df[df["set_name"].astype(str) == str(set_name)]
    return _split(hit.iloc[0]["members"]) if not hit.empty else []


def disease_state_rules_df() -> pd.DataFrame:
    """Frozen declarative disease-state rule compatibility snapshot.

    New per-sample disease-state interpretation belongs to trufflepig. Legacy
    columns include ``rule_id``, ``cancer_code``, ``claims``, ``conditions``, and
    ``narrative``. Defensive copy.
    """
    warn_legacy_dataset_access("disease-state-rules", stacklevel=2)
    return get_data("disease-state-rules").copy()


def degenerate_subtype_pairs_df() -> pd.DataFrame:
    """Expression-degenerate subtype pairs + their tiebreaker rules. Defensive copy."""
    return get_data("degenerate-subtype-pairs").copy()
