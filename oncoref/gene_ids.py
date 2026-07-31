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

"""Gene-id and symbol resolution references.

oncoref data comes from many pipelines on different Ensembl/HGNC bases. These
curated maps reconcile them: alt-haplotype Ensembl gene ids → their primary-contig
id, NCBI symbol synonyms → the official symbol, plus supplemental transcript→gene
mappings and the cDNA-identical / proteoform-collapse-override grouping refinements.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .load_dataset import _register_derived_cache, get_data


def unversioned(gene_id: str) -> str:
    """Canonical Ensembl-id normalizer: ``ENSG00000251562.5`` → ``ENSG00000251562``,
    stripping a single version suffix *and* surrounding whitespace (idempotent on bare
    ids). The one definition every layer shares, so a padded/versioned id can't match
    in one place and miss in another."""
    return str(gene_id).split(".", 1)[0].strip()


#: Back-compat internal alias.
_unversioned = unversioned


@lru_cache(maxsize=1)
def ensembl_id_aliases() -> dict[str, str]:
    """``{alt_haplotype_gene_id: primary_contig_gene_id}`` (unversioned) — alt-locus
    Ensembl gene ids mapped to their primary-assembly id."""
    df = get_data("ensembl-id-aliases", copy=False)
    return {
        _unversioned(a): _unversioned(p)
        for a, p in zip(df["alt_haplotype_id"], df["primary_contig_id"])
    }


def resolve_ensembl_id(gene_id: str) -> str:
    """Map an Ensembl gene id to its canonical primary-assembly id (unversioned),
    resolving alt-haplotype / patch copies and cross-release ID turnover through the
    shipped ensembl-id-aliases map; returns the id unchanged if it isn't an alias."""
    key = _unversioned(gene_id)
    if key.upper().startswith("ENSG"):
        key = key.upper()
    return ensembl_id_aliases().get(key, key)


def canonical_gene_id(identifier: str, *, source_version: str | None = None) -> str | None:
    """Map **any** gene identifier into the harmonized canonical ENSG space.

    The single entry point a consumer reaches for when harmonizing arbitrary inputs
    (oncoref#135): accepts an Ensembl gene id from *any* release/assembly — versioned or
    not, alt-haplotype, patch, or a retired/old-release id — or an HGNC/NCBI gene
    **symbol**, and returns the canonical (newest primary-assembly) ENSG, or ``None`` if
    it can't be resolved.

    - An ``ENSG…`` id is unversioned and run through the alias / cross-release migration
      map (:func:`resolve_ensembl_id`) — so e.g. GRCh37 ``GGNBP2`` ``ENSG00000005955`` →
      ``ENSG00000278311``. The resolved id is returned only if it is in oncoref's
      canonical gene space; unknown Ensembl-like ids return ``None``.
    - A numeric identifier is treated as an Entrez / NCBI GeneID and resolved through
      the shipped NCBI-derived Entrez→canonical ENSG map.
    - Anything else is treated as a symbol and resolved via the installed Ensembl
      releases + NCBI synonyms (lazy import of the genome layer; needs the ``genome``
      extra). Symbol resolution already returns a canonical primary-assembly id.

    ``source_version`` (e.g. ``"GRCh37"`` / ``"75"``) is accepted for forward
    compatibility and caller intent; the migration map is release-agnostic today."""
    s = str(identifier).strip()
    if not s:
        return None
    if s.upper().startswith("ENSG"):
        resolved = resolve_ensembl_id(s)
        return resolved if resolved in _canonical_gene_index() else None
    if s.isdigit():
        hit = resolve_entrez_id(s)
        return hit[0] if hit is not None else None
    symbol_index = _canonical_symbol_index()
    direct = symbol_index.get(s.upper())
    if direct is not None:
        return direct
    official = resolve_symbol(s)
    resolved = symbol_index.get(official.upper())
    if resolved is not None:
        return resolved
    from .genome import GenomeDependencyError, canonical_gene_id_and_name

    try:
        gid, _ = canonical_gene_id_and_name(official)
    except GenomeDependencyError:
        return None
    return gid


def canonical_gene_ids(
    identifiers: list[str], *, source_version: str | None = None
) -> list[str | None]:
    """Batch :func:`canonical_gene_id` → one canonical ENSG (or ``None``) per input."""
    return [canonical_gene_id(x, source_version=source_version) for x in identifiers]


def canonical_gene_symbol(identifier: str, *, source_version: str | None = None) -> str | None:
    """Map any supported gene identifier to its canonical display symbol.

    This is the symbol companion to :func:`canonical_gene_id`: Ensembl ids are
    migrated through the alias/cross-release table, and symbols or synonyms are
    resolved through the same any-identifier path. Returns ``None`` when the input
    cannot be mapped into oncoref's canonical gene space.
    """
    gid = canonical_gene_id(identifier, source_version=source_version)
    if gid is None:
        return None
    hit = _canonical_gene_index().get(resolve_ensembl_id(gid))
    return hit[0] if hit else None


def canonical_gene_symbols(
    identifiers: list[str], *, source_version: str | None = None
) -> list[str | None]:
    """Batch :func:`canonical_gene_symbol` → one canonical symbol (or ``None``) per input."""
    return [canonical_gene_symbol(x, source_version=source_version) for x in identifiers]


def entrez_gene_mappings() -> pd.DataFrame:
    """NCBI Entrez / GeneID mappings into oncoref's canonical gene space.

    The wheel ships a filtered table built from pinned NCBI ``gene_info`` and
    ``gene_history`` snapshots: live Entrez IDs resolve through direct Ensembl dbXrefs
    when possible, otherwise through the current NCBI symbol; discontinued Entrez IDs
    resolve through their live replacement. The raw NCBI files are intentionally not
    shipped in the wheel.
    """
    out = get_data("ncbi-entrez-gene-mappings").copy()
    for col in (
        "entrez_id",
        "live_entrez_id",
        "canonical_ensembl_gene_id",
        "canonical_symbol",
        "mapping_method",
        "current_mapping_method",
    ):
        out[col] = out[col].astype(str)
    return out


@lru_cache(maxsize=1)
def _entrez_gene_index() -> dict[str, tuple[str, str, str]]:
    df = get_data("ncbi-entrez-gene-mappings", copy=False)
    return {
        str(entrez_id): (str(gene_id), str(symbol), str(method))
        for entrez_id, gene_id, symbol, method in zip(
            df["entrez_id"],
            df["canonical_ensembl_gene_id"],
            df["canonical_symbol"],
            df["mapping_method"],
        )
    }


def resolve_entrez_id(entrez_id: str) -> tuple[str, str, str] | None:
    """Map an Entrez / NCBI GeneID to ``(canonical_ensg, symbol, method)``.

    ``method`` records direct Ensembl dbXrefs, current-symbol resolution, the
    explicit NCBI-to-Ensembl mitochondrial-symbol translation, or GeneID history.
    Returns ``None`` when the ID is blank, non-numeric, or absent from the shipped
    NCBI-derived mapping table.
    """
    raw = str(entrez_id).strip()
    if not raw or not raw.isdigit():
        return None
    return _entrez_gene_index().get(raw)


def _looks_like_ensembl_gene_id(identifier: str) -> bool:
    return unversioned(identifier).upper().startswith("ENSG")


def display_gene_name(
    identifier: str | None, *, source_version: str | None = None, fallback: bool = True
) -> str | None:
    """Report-facing gene label for an arbitrary identifier.

    Prefer the canonical oncoref symbol. If a non-Ensembl text label cannot be
    resolved and ``fallback=True``, return the stripped input after synonym cleanup
    so reports can still show the caller's label. Unknown Ensembl ids return
    ``None`` rather than pretending the id is a gene symbol.
    """
    if identifier is None:
        return None
    raw = str(identifier).strip()
    if not raw:
        return None
    symbol = canonical_gene_symbol(raw, source_version=source_version)
    if symbol is not None:
        return symbol
    if not fallback or _looks_like_ensembl_gene_id(raw):
        return None
    return resolve_symbol(raw).strip() or None


def short_gene_name(
    identifier: str | None, *, source_version: str | None = None, fallback: bool = True
) -> str | None:
    """Compact report label for a gene identifier.

    Currently this is the same stable symbol chosen by :func:`display_gene_name`.
    It exists as an explicit public hook so downstream report code does not invent
    its own resolver or silently bypass oncoref's canonical gene space.
    """
    return display_gene_name(identifier, source_version=source_version, fallback=fallback)


def canonical_gene_space() -> pd.DataFrame:
    """The authoritative **canonical gene-ID space** (oncoref#135 item 4): one row per
    primary-assembly Ensembl gene — ``ensembl_gene_id``, ``symbol``, ``biotype``,
    ``seqname``, ``ensembl_release`` — that the alias/migration map resolves *into*.
    Filter on ``biotype == "protein_coding"`` to drop the large ncRNA/pseudogene tail.
    Defensive copy."""
    return get_data("canonical-gene-space").copy()


@lru_cache(maxsize=1)
def _canonical_gene_index() -> dict[str, tuple[str, str]]:
    """``{ensembl_gene_id (unversioned): (symbol, biotype)}`` over the canonical space."""
    df = get_data("canonical-gene-space", copy=False)
    return {
        _unversioned(str(g)): (str(s), str(b))
        for g, s, b in zip(df["ensembl_gene_id"], df["symbol"], df["biotype"])
    }


@lru_cache(maxsize=1)
def _canonical_symbol_index() -> dict[str, str]:
    """``{official_symbol.upper(): canonical_ensembl_gene_id}`` over canonical genes."""
    df = get_data("canonical-gene-space", copy=False)
    out: dict[str, str] = {}
    for gid, symbol in zip(df["ensembl_gene_id"], df["symbol"]):
        sym = str(symbol).strip()
        if sym:
            out.setdefault(sym.upper(), _unversioned(str(gid)))
    return out


def is_canonical_gene(gene_id: str) -> bool:
    """Is ``gene_id`` (after alias/migration resolution) a member of the canonical
    primary-assembly gene space? ``False`` for RefSeq-lift / non-Ensembl / retired-without-
    successor ids that have no canonical entry."""
    return resolve_ensembl_id(gene_id) in _canonical_gene_index()


def gene_biotype(gene_id: str) -> str | None:
    """Ensembl biotype (``"protein_coding"``, ``"lncRNA"``, …) of the canonical gene
    ``gene_id`` resolves to, or ``None`` if it isn't in the canonical space."""
    hit = _canonical_gene_index().get(resolve_ensembl_id(gene_id))
    return hit[1] if hit else None


def is_protein_coding_gene(gene_id: str) -> bool:
    """``True`` iff ``gene_id`` resolves to a ``protein_coding`` canonical gene."""
    return gene_biotype(gene_id) == "protein_coding"


@lru_cache(maxsize=1)
def ensembl_id_alias_symbols() -> dict[str, str]:
    """``{primary_contig_gene_id (unversioned): gene_symbol}`` — the curated canonical
    symbol for a migrated/consolidated locus, from the alias table (rows with no symbol,
    e.g. archive replacements, are skipped)."""
    df = get_data("ensembl-id-aliases", copy=False)
    out: dict[str, str] = {}
    for primary, symbol in zip(df["primary_contig_id"], df["symbol"]):
        sym = str(symbol).strip()
        if sym and sym.lower() != "nan":
            out[_unversioned(str(primary))] = sym
    return out


@lru_cache(maxsize=1)
def _symbol_synonym_indexes() -> tuple[dict[str, str], dict[str, str]]:
    """Build exact-case and safe case-insensitive NCBI synonym indexes."""
    df = get_data("ncbi-symbol-synonyms", copy=False)
    exact: dict[str, str] = {}
    targets_by_uppercase_alias: dict[str, set[str]] = {}
    for alias_value, official_value in zip(df["alias"], df["official_symbol"]):
        if pd.isna(alias_value) or pd.isna(official_value):
            continue
        alias = str(alias_value)
        official = str(official_value)
        if not alias or not official:
            continue
        previous = exact.get(alias)
        if previous is not None and previous != official:
            raise ValueError(
                f"NCBI synonym alias {alias!r} maps to both {previous!r} and {official!r}"
            )
        exact[alias] = official
        targets_by_uppercase_alias.setdefault(alias.upper(), set()).add(official)

    case_insensitive: dict[str, str] = {}
    for uppercase_alias, targets in targets_by_uppercase_alias.items():
        if len(targets) == 1:
            case_insensitive[uppercase_alias] = next(iter(targets))
        elif uppercase_alias in exact:
            # The snapshot's exact uppercase row is the deterministic fallback.
            case_insensitive[uppercase_alias] = exact[uppercase_alias]
    return exact, case_insensitive


_register_derived_cache(_symbol_synonym_indexes.cache_clear)


def symbol_synonyms() -> dict[str, str]:
    """Safe case-insensitive ``{ALIAS: official_symbol}`` NCBI synonym index.

    Keys are uppercase. Ambiguous case-folded aliases are omitted unless the
    NCBI snapshot contains an exact uppercase alias, which supplies the stable
    fallback. Use :func:`resolve_symbol` when exact-case distinctions matter.
    """
    return _symbol_synonym_indexes()[1].copy()


def resolve_symbol(symbol: str) -> str:
    """Resolve an NCBI gene-symbol synonym without collapsing case collisions.

    Exact-case aliases win. Other casing uses the safe case-insensitive index;
    an ambiguous fold with no exact uppercase row is returned unchanged.
    """
    raw = str(symbol)
    exact, case_insensitive = _symbol_synonym_indexes()
    exact_match = exact.get(raw)
    if exact_match is not None:
        return exact_match
    return case_insensitive.get(raw.upper(), raw)


def gene_identifier_mapping_coverage() -> pd.DataFrame:
    """Coverage report for oncoref's shipped gene identifier mappings.

    One row per canonical gene-space entry, with explicit flags for whether its
    canonical ENSG and symbol round-trip through the public resolver, whether any
    prior/alt Ensembl ids resolve to it, and whether the NCBI synonym table carries
    aliases for its current symbol. Lack of aliases is not itself an error; this
    table makes those boundaries visible for downstream parity audits (#279).
    """
    out = canonical_gene_space()
    out["ensembl_gene_id"] = out["ensembl_gene_id"].map(_unversioned)
    out["symbol"] = out["symbol"].fillna("").astype(str).str.strip()
    out["has_symbol"] = out["symbol"].ne("")
    out["_symbol_key"] = out["symbol"].str.upper()

    symbol_index = _canonical_symbol_index()
    out["canonical_id_roundtrip"] = (
        out["ensembl_gene_id"].map(resolve_ensembl_id).eq(out["ensembl_gene_id"])
    )
    symbol_counts = out.loc[out["has_symbol"], "_symbol_key"].value_counts().to_dict()
    out["n_genes_with_symbol"] = [int(symbol_counts.get(sym, 0)) for sym in out["_symbol_key"]]
    out["symbol_resolves_to_ensembl_gene_id"] = [
        symbol_index.get(sym) if sym else None for sym in out["_symbol_key"]
    ]

    def _symbol_issue(row) -> str:
        if not row.has_symbol:
            return "missing_symbol"
        if row.n_genes_with_symbol > 1:
            return "non_unique_symbol"
        if row.symbol_resolves_to_ensembl_gene_id is None:
            return "symbol_not_resolvable"
        if row.symbol_resolves_to_ensembl_gene_id != row.ensembl_gene_id:
            return "symbol_resolves_to_other_gene"
        return "ok"

    out["symbol_roundtrip_issue"] = [_symbol_issue(row) for row in out.itertuples(index=False)]
    out["symbol_roundtrip"] = out["symbol_roundtrip_issue"].eq("ok")

    syn_df = get_data("ncbi-symbol-synonyms", copy=False)
    entrez_df = get_data("ncbi-entrez-gene-mappings", copy=False)
    synonym_counts = (
        syn_df.assign(_official=syn_df["official_symbol"].astype(str).str.upper())
        .groupby("_official", sort=False)
        .size()
        .to_dict()
    )
    out["n_symbol_aliases"] = [int(synonym_counts.get(sym.upper(), 0)) for sym in out["symbol"]]
    out["has_symbol_alias"] = out["n_symbol_aliases"].gt(0)

    alias_counts: dict[str, int] = {}
    for primary in ensembl_id_aliases().values():
        alias_counts[primary] = alias_counts.get(primary, 0) + 1
    out["n_ensembl_aliases"] = [int(alias_counts.get(gid, 0)) for gid in out["ensembl_gene_id"]]
    out["has_ensembl_alias"] = out["n_ensembl_aliases"].gt(0)

    entrez_counts = entrez_df.groupby("canonical_ensembl_gene_id", sort=False).size().to_dict()
    out["n_entrez_ids"] = [int(entrez_counts.get(gid, 0)) for gid in out["ensembl_gene_id"]]
    out["has_entrez_id"] = out["n_entrez_ids"].gt(0)

    def _status(row) -> str:
        if not row.has_symbol:
            return "missing_symbol"
        if not row.canonical_id_roundtrip:
            return "canonical_id_roundtrip_failed"
        if row.symbol_roundtrip_issue != "ok":
            return "symbol_roundtrip_failed"
        return "ok"

    out["mapping_status"] = [_status(row) for row in out.itertuples(index=False)]
    return out[
        [
            "ensembl_gene_id",
            "symbol",
            "biotype",
            "seqname",
            "ensembl_release",
            "has_symbol",
            "canonical_id_roundtrip",
            "symbol_roundtrip",
            "symbol_roundtrip_issue",
            "n_genes_with_symbol",
            "symbol_resolves_to_ensembl_gene_id",
            "n_symbol_aliases",
            "has_symbol_alias",
            "n_ensembl_aliases",
            "has_ensembl_alias",
            "n_entrez_ids",
            "has_entrez_id",
            "mapping_status",
        ]
    ]


def gene_identifier_mapping_summary() -> pd.DataFrame:
    """One-row summary of :func:`gene_identifier_mapping_coverage`."""
    coverage = gene_identifier_mapping_coverage()
    syn_df = get_data("ncbi-symbol-synonyms", copy=False)
    entrez_df = get_data("ncbi-entrez-gene-mappings", copy=False)
    alias_map = ensembl_id_aliases()
    return pd.DataFrame(
        [
            {
                "n_genes": len(coverage),
                "n_protein_coding": int(coverage["biotype"].eq("protein_coding").sum()),
                "n_without_symbol": int((~coverage["has_symbol"]).sum()),
                "n_canonical_id_roundtrip_failed": int((~coverage["canonical_id_roundtrip"]).sum()),
                "n_symbol_roundtrip_failed": int(
                    coverage["mapping_status"].eq("symbol_roundtrip_failed").sum()
                ),
                "n_non_unique_symbols": int(
                    coverage["symbol_roundtrip_issue"].eq("non_unique_symbol").sum()
                ),
                "n_symbol_not_resolvable": int(
                    coverage["symbol_roundtrip_issue"].eq("symbol_not_resolvable").sum()
                ),
                "n_symbol_resolves_to_other_gene": int(
                    coverage["symbol_roundtrip_issue"].eq("symbol_resolves_to_other_gene").sum()
                ),
                "n_with_symbol_aliases": int(coverage["has_symbol_alias"].sum()),
                "n_with_ensembl_aliases": int(coverage["has_ensembl_alias"].sum()),
                "n_with_entrez_ids": int(coverage["has_entrez_id"].sum()),
                "n_symbol_alias_rows": len(syn_df),
                "n_ensembl_alias_rows": len(alias_map),
                "n_entrez_mapping_rows": len(entrez_df),
                "mapping_statuses": ";".join(sorted(coverage["mapping_status"].unique())),
                "symbol_roundtrip_issues": ";".join(
                    sorted(coverage["symbol_roundtrip_issue"].unique())
                ),
            }
        ]
    )


def extra_transcript_mappings() -> pd.DataFrame:
    """Supplemental transcript→gene mappings (``transcript_id``, ``gene_symbol``,
    ``ensembl_gene_id``, ``biotype``, …). Defensive copy."""
    return get_data("extra-tx-mappings").copy()


def cdna_identical_groups() -> pd.DataFrame:
    """cDNA-identical gene groups (one row per member: ``group_canonical_*``,
    ``ensembl_gene_id``, ``symbol``, ``cds_nt``, ``n_members``). Defensive copy."""
    return get_data("cdna-identical-gene-groups").copy()


def proteoform_collapse_overrides() -> pd.DataFrame:
    """Manual proteoform-collapse overrides (``group_canonical_ensembl_gene_id``,
    ``group_symbol``, ``reason``). Defensive copy."""
    return get_data("proteoform-collapse-overrides").copy()
