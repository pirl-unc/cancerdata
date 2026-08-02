# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the pure gene-ranking helpers (no Ensembl release needed)."""

from types import SimpleNamespace

import pytest

from oncoref import genome as gx


class _Tx:
    def __init__(self, *, support_level=None, is_protein_coding=True, start=True, cds="A" * 30):
        self.support_level = support_level
        self.is_protein_coding = is_protein_coding
        self.contains_start_codon = start
        self.coding_sequence = cds


class _Gene:
    def __init__(self, name, transcripts, biotype="protein_coding"):
        self.name = name
        self.transcripts = transcripts
        self.biotype = biotype


@pytest.fixture(autouse=True)
def _clear_id_resolver_caches():
    resolvers = (
        gx.find_gene_name_from_ensembl_gene_id,
        gx.find_gene_name_from_ensembl_transcript_id,
    )
    for resolver in resolvers:
        resolver.cache_clear()
    yield
    for resolver in resolvers:
        resolver.cache_clear()


def test_id_resolvers_reject_malformed_ids_without_opening_a_genome(monkeypatch):
    monkeypatch.setattr(
        gx, "genomes", lambda: pytest.fail("malformed IDs must not query pyensembl")
    )

    assert gx.find_gene_name_from_ensembl_gene_id("ENSG_FAKE") is None
    assert gx.find_gene_name_from_ensembl_transcript_id("ENST_FAKE") is None


def test_transcript_resolver_queries_releases_directly_and_caches(monkeypatch):
    calls = []

    class Genome:
        def __init__(self, gene_name=None):
            self.gene_name = gene_name

        def transcript_by_id(self, transcript_id):
            calls.append((self.gene_name, transcript_id))
            if self.gene_name is None:
                raise ValueError(transcript_id)
            return SimpleNamespace(gene_name=self.gene_name)

    monkeypatch.setattr(gx, "genomes", lambda: [Genome(), Genome("TP53")])

    assert gx.find_gene_name_from_ensembl_transcript_id("ENST00000269305.9") == "TP53"
    assert gx.find_gene_name_from_ensembl_transcript_id("ENST00000269305.9") == "TP53"
    assert calls == [(None, "ENST00000269305"), ("TP53", "ENST00000269305")]


def test_gene_resolver_queries_exact_ids_without_materializing_release(monkeypatch):
    genome = SimpleNamespace(
        gene_by_id=lambda gene_id: SimpleNamespace(gene_name=None, name="TP53")
    )
    monkeypatch.setattr(gx, "genomes", lambda: [genome])

    assert gx.find_gene_name_from_ensembl_gene_id("ENSG00000141510") == "TP53"


def test_best_transcript_support_prefers_lower_tsl():
    assert gx._best_transcript_support(_Gene("G", [_Tx(support_level=1)])) == -1
    assert gx._best_transcript_support(_Gene("G", [_Tx(support_level=5)])) == -5
    # best (lowest) TSL wins among several
    assert (
        gx._best_transcript_support(_Gene("G", [_Tx(support_level=3), _Tx(support_level=1)])) == -1
    )


def test_no_tsl_info_ranks_worst_not_zero():
    # The bug: a gene with no TSL info used to score 0 and outrank a clean TSL-1 gene.
    no_info = _Gene("G", [_Tx(support_level=None), _Tx(support_level="NA")])
    assert gx._best_transcript_support(no_info) == gx._NO_TSL_SCORE
    assert gx._NO_TSL_SCORE < -5  # below even a TSL-5 transcript


def test_pick_best_gene_assessed_beats_unassessed():
    good = _Gene("ASSESSED", [_Tx(support_level=1)])
    unknown = _Gene("UNKNOWN", [_Tx(support_level=None)])
    assert gx.pick_best_gene([unknown, good]) is good
    assert gx.pick_best_gene([good, unknown]) is good


def test_best_cds_length_tolerates_missing_coding_sequence():
    # A protein-coding transcript with no assembled CDS (GTF-only install) must not crash.
    g = _Gene("G", [_Tx(cds=None), _Tx(cds="A" * 60)])
    assert gx._best_canonical_cds_length(g) == 60
    # all-missing -> 0, not a TypeError
    assert gx._best_canonical_cds_length(_Gene("G", [_Tx(cds=None)])) == 0


def test_pick_best_gene_single_and_empty():
    only = _Gene("X", [_Tx(support_level=1)])
    assert gx.pick_best_gene([only]) is only
    with pytest.raises(ValueError, match="at least one gene"):
        gx.pick_best_gene([])
