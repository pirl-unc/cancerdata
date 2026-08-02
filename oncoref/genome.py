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

"""Ensembl-reference gene/transcript resolution (the ``pyensembl``-backed layer).

This is the genome-reference resolver that the curated pandas ID layer
(:mod:`oncoref.gene_ids`, curated CSV aliases) can't do: mapping an *arbitrary*
Ensembl transcript or gene ID to a gene, and a symbol to its canonical Ensembl gene
ID, against the installed Ensembl release(s).

Install the optional ``genome`` extra to use this module (``pip install
'oncoref[genome]'``). pyensembl still needs a **downloaded** human Ensembl release at
runtime (``pyensembl install --release N --species homo_sapiens``); with the extra but
no release installed, :func:`genomes` is empty and the resolvers return ``None``.

Resolution order for a symbol mirrors pirlygenes: each installed release (newest
first) by name + curated display aliases, then the bundled NCBI symbol-synonym
snapshot (:func:`oncoref.gene_ids.resolve_symbol`). Gene/transcript IDs resolve
against the newest release first, falling back to older installed releases.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from .gene_ids import resolve_symbol, unversioned


class GenomeDependencyError(ImportError):
    """Raised when a pyensembl-backed API is used without the genome extra."""


_GENOME_EXTRA_MESSAGE = (
    "pyensembl is required for genome and peptide APIs; "
    "install it with `pip install 'oncoref[genome]'`"
)

#: Curated literary display aliases used as extra symbol candidates (NY-ESO-1 ↔
#: CTAG1B, gp100 ↔ PMEL, …). Small, base-layer reference — not a full alias table.
_DISPLAY_ALIASES = {
    "CTAG1B": "NY-ESO-1",
    "PMEL": "gp100",
    "CD274": "PD-L1",
    "PDCD1LG2": "PD-L2",
    "PDCD1": "PD-1",
    "FOLH1": "PSMA",
    "TNFRSF17": "BCMA",
    "TACSTD2": "TROP2",
    "MS4A1": "CD20",
    "HAVCR2": "TIM-3",
    "VSIR": "VISTA",
}
_REVERSE_ALIASES: dict[str, list[str]] = {}
for _k, _v in _DISPLAY_ALIASES.items():
    _REVERSE_ALIASES.setdefault(_v, []).append(_k)


def _restore_root_logger(root, handlers, level, disabled) -> None:
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)
    root.disabled = disabled


@lru_cache(maxsize=1)
def _installed_ensembl_releases():
    """Collect pyensembl releases without inheriting gtfparse root logging setup."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    disabled = root.disabled
    try:
        try:
            from pyensembl.shell import collect_all_installed_ensembl_releases
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".", 1)[0] == "pyensembl":
                raise GenomeDependencyError(_GENOME_EXTRA_MESSAGE) from exc
            raise

        return tuple(collect_all_installed_ensembl_releases())
    finally:
        _restore_root_logger(root, handlers, level, disabled)


@lru_cache(maxsize=1)
def genomes():
    """Installed human Ensembl releases, newest first. Empty if none installed."""
    return sorted(
        (g for g in _installed_ensembl_releases() if g.species.latin_name == "homo_sapiens"),
        key=lambda g: g.release,
        reverse=True,
    )


def strip_version(gene_id: str) -> str:
    """``ENSG00000251562.5`` → ``ENSG00000251562`` (idempotent on bare ids). Alias of
    :func:`oncoref.gene_ids.unversioned`, the one shared normalizer."""
    return unversioned(gene_id)


def gene_for_ensembl_id(genome, gene_id: str):
    """Resolve an Ensembl gene id to a pyensembl ``Gene`` against one release, or
    ``None`` (version-tolerant, exception-swallowing)."""
    try:
        return genome.gene_by_id(strip_version(gene_id))
    except Exception:
        return None


_ENSEMBL_GENE_ID = re.compile(r"ENSG\d+")
_ENSEMBL_TRANSCRIPT_ID = re.compile(r"ENST\d+")


@lru_cache(maxsize=65_536)
def find_gene_name_from_ensembl_gene_id(gene_id: str) -> str | None:
    """Gene symbol for a human Ensembl gene id, newest installed release first."""
    gid = strip_version(gene_id)
    if _ENSEMBL_GENE_ID.fullmatch(gid) is None:
        return None
    for genome in genomes():
        gene = gene_for_ensembl_id(genome, gid)
        if gene:
            return getattr(gene, "gene_name", None) or getattr(gene, "name", None)
    return None


@lru_cache(maxsize=65_536)
def find_gene_name_from_ensembl_transcript_id(transcript_id: str) -> str | None:
    """Gene symbol for a human Ensembl transcript id, newest installed release first."""
    tid = strip_version(transcript_id)
    if _ENSEMBL_TRANSCRIPT_ID.fullmatch(tid) is None:
        return None
    for genome in genomes():
        try:
            t = genome.transcript_by_id(tid)
        except Exception:
            t = None
        if t and t.gene_name:
            return t.gene_name
    return None


def _best_canonical_cds_length(gene) -> int:
    lengths = []
    for t in gene.transcripts:
        if not (t.is_protein_coding and t.contains_start_codon):
            continue
        # A protein-coding transcript can still lack an assembled coding sequence when
        # only the GTF is installed (no cDNA FASTA) — len(None) would crash resolution.
        cds = getattr(t, "coding_sequence", None)
        if cds is not None:
            lengths.append(len(cds))
    return max(lengths, default=0)


#: Score for a gene whose transcripts carry no TSL information at all. Ensembl TSL is
#: 1 (best)..5 (worst); -min(TSL) lands in [-5, -1], so a sortable score strictly
#: below -5 makes "no support info" rank *below* even a TSL-5 transcript (rather than
#: above a clean TSL-1 one, which `0` did — a real mis-ranking in pick_best_gene).
_NO_TSL_SCORE = -6


def _best_transcript_support(gene) -> int:
    """Quality of the gene's best transcript as a sortable score (higher better):
    Ensembl TSL is 1 (best)..5 (worst), inverted to ``-min(TSL)``; transcripts with
    no/NA TSL are skipped, and a gene with *no* TSL info scores :data:`_NO_TSL_SCORE`
    (worst), so an un-assessed gene never outranks an assessed one."""
    levels = []
    for t in gene.transcripts:
        try:
            levels.append(int(getattr(t, "support_level", None)))
        except (TypeError, ValueError):
            continue
    return -min(levels) if levels else _NO_TSL_SCORE


def pick_best_gene(genes):
    """Choose one gene when a symbol maps to several (pirlygenes order): prefer
    protein-coding, then higher-quality best transcript (lowest TSL), then longer
    canonical CDS, more coding transcripts, then a stable name tie-break. A symbol
    resolves to ONE Ensembl gene id — the only key used for joins downstream."""
    if not genes:
        raise ValueError("expected at least one gene")
    if len(genes) == 1:
        return genes[0]

    def sort_key(g):
        coding = 1 if getattr(g, "biotype", "") == "protein_coding" else 0
        num_coding = sum(t.is_protein_coding for t in g.transcripts)
        return (
            coding,
            _best_transcript_support(g),
            _best_canonical_cds_length(g),
            num_coding,
            -g.name.count("."),
            len(g.name),
            g.name,
        )

    return sorted(genes, key=sort_key, reverse=True)[0]


def _name_candidates(name: str) -> set[str]:
    cands = {name, *(_DISPLAY_ALIASES.get(name, name),), *_REVERSE_ALIASES.get(name, [])}
    return {c for n in list(cands) for c in (n, n.lower(), n.upper())}


def find_gene_and_release_by_name(name: str):
    """``(genome, Gene)`` for a symbol — each release (newest first) by name +
    curated aliases, then the NCBI synonym fallback. ``None`` if unresolved."""
    for genome in genomes():
        for n in _name_candidates(name):
            try:
                genes = genome.genes_by_name(n)
            except Exception:
                genes = []
            if genes:
                return genome, pick_best_gene(genes)
    official = resolve_symbol(name)  # bundled NCBI symbol-synonym snapshot
    if official and official != name:
        for genome in genomes():
            try:
                genes = genome.genes_by_name(official)
            except Exception:
                genes = []
            if genes:
                return genome, pick_best_gene(genes)
    return None


def find_gene_id_by_name(name: str) -> str | None:
    """Canonical Ensembl gene id for a symbol, or ``None``."""
    res = find_gene_and_release_by_name(name)
    return res[1].id if res else None


def canonical_gene_id_and_name(name: str) -> tuple[str | None, str | None]:
    """``(Ensembl gene id, canonical symbol)`` for a symbol, or ``(None, None)``."""
    res = find_gene_and_release_by_name(name)
    return (res[1].id, res[1].name) if res else (None, None)


def canonical_gene_ids_and_names(names):
    """Batch :func:`canonical_gene_id_and_name` → ``([ids], [names])``."""
    ids, syms = [], []
    for name in names:
        gid, sym = canonical_gene_id_and_name(name)
        ids.append(gid)
        syms.append(sym)
    return ids, syms


def aggregate_gene_expression(df, tx_to_gene_name=None, **kwargs):
    """Full transcript→gene aggregation with Ensembl-reference resolution.

    Like :func:`oncoref.expression_engine.aggregate_transcripts_to_genes`, but
    resolves transcripts NOT in ``tx_to_gene_name`` against the installed Ensembl
    release(s) (via :func:`find_gene_name_from_ensembl_transcript_id`) instead of
    bucketing them as ``unresolved`` — the faithful drop-in for pirlygenes'
    ``aggregate_gene_expression``. Adds ``gene_id`` (canonical Ensembl id) per gene.
    Requires the genome extra; pass extra ``aggregate_transcripts_to_genes`` kwargs
    through (column candidates, etc.)."""
    from .expression_engine import aggregate_transcripts_to_genes, expanded_tx_map, find_column

    tx_col = find_column(
        df,
        kwargs.get(
            "transcript_id_column_candidates", ("transcript", "transcript_id", "target_id", "name")
        ),
        "transcript ID",
    )
    base_map = dict(tx_to_gene_name) if tx_to_gene_name is not None else {}
    expanded = expanded_tx_map(base_map)
    # Resolve the transcripts the base map doesn't cover, via the genome.
    tx0 = df[tx_col].astype(str).str.split(".", n=1).str[0]
    resolved = dict(base_map)
    for tx in tx0[~tx0.isin(expanded)].unique():
        name = find_gene_name_from_ensembl_transcript_id(tx)
        if name:
            resolved[tx] = name
    agg = aggregate_transcripts_to_genes(df, resolved, **kwargs)
    # Attach canonical Ensembl gene ids for the resolved genes.
    real = agg["gene"] != "unresolved"
    id_map = {g: find_gene_id_by_name(g) for g in agg.loc[real, "gene"].unique()}
    agg["gene_id"] = agg["gene"].map(lambda g: id_map.get(g))
    return agg
