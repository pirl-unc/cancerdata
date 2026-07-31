# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Gene-id / symbol resolution references (#35, R-resolve)."""

import oncoref
from oncoref import gene_ids as g
from oncoref.load_dataset import get_data


def test_unversioned_normalizer():
    assert g.unversioned("ENSG00000141510.5") == "ENSG00000141510"
    assert g.unversioned("ENSG00000141510") == "ENSG00000141510"  # idempotent on bare
    assert g.unversioned("  ENSG00000141510.5  ") == "ENSG00000141510"  # strips whitespace
    assert g.unversioned("ENSG00000141510") == g.unversioned(" ENSG00000141510")  # padded == bare


def test_ensembl_id_alias_resolution():
    aliases = g.ensembl_id_aliases()
    assert aliases  # non-empty
    alt, primary = next(iter(aliases.items()))
    assert g.resolve_ensembl_id(alt) == primary
    assert g.resolve_ensembl_id(f"{alt}.7") == primary  # version-insensitive
    # a non-alias id passes through unchanged (unversioned)
    assert g.resolve_ensembl_id("ENSG99999999999.3") == "ENSG99999999999"


def test_alias_table_is_migration_aware_and_acyclic():
    # The map covers both alt-haplotype/patch copies and cross-release id turnover, and
    # must be a clean forest: no self-maps, no duplicate keys, and no chains (a target
    # that is itself an alias key) so a single lookup always lands on a canonical id.
    aliases = g.ensembl_id_aliases()
    keys, targets = set(aliases), set(aliases.values())
    assert not (keys & targets), "alias targets must not themselves be alias keys (no chains)"
    assert all(k != v for k, v in aliases.items()), "no self-maps"
    # cross-release migration: GRCh37 GGNBP2 retired -> its current primary-assembly id
    assert g.resolve_ensembl_id("ENSG00000005955") == "ENSG00000278311"
    # resolution is idempotent (targets are canonical, never themselves keys)
    for k in list(aliases)[:2000]:
        assert g.resolve_ensembl_id(g.resolve_ensembl_id(k)) == g.resolve_ensembl_id(k)


def test_every_alias_target_is_in_the_canonical_space():
    # resolve_ensembl_id must always land in the canonical gene space, so a resolved id
    # has a biotype and passes is_canonical_gene — otherwise the migration map and the
    # space artifact disagree about what "canonical" means.
    space = set(g.canonical_gene_space()["ensembl_gene_id"])
    targets = set(g.ensembl_id_aliases().values())
    assert targets <= space, sorted(targets - space)[:5]


def test_canonical_gene_id_any_identifier():
    # The unified entry point (oncoref#135 item 1): any Ensembl id -> canonical ENSG.
    assert g.canonical_gene_id("ENSG00000005955") == "ENSG00000278311"  # old GRCh37 id
    assert g.canonical_gene_id("ENSG00000005955.7") == "ENSG00000278311"  # version-insensitive
    assert g.canonical_gene_id("ENSG00000278311") == "ENSG00000278311"  # already canonical
    assert g.canonical_gene_id("ensg00000141510.17") == "ENSG00000141510"  # case-insensitive
    assert g.canonical_gene_id("TP53") == "ENSG00000141510"  # direct canonical symbol
    assert g.canonical_gene_id("GNB2L1") == "ENSG00000204628"  # prior symbol -> RACK1
    assert g.canonical_gene_id("TCEB2") == "ENSG00000103363"  # prior symbol -> ELOB
    assert g.canonical_gene_id("7157") == "ENSG00000141510"  # Entrez/GeneID -> TP53
    assert g.canonical_gene_id("ENSG99999999999") is None  # unknown ENSG is not canonicalized
    assert g.canonical_gene_id("") is None and g.canonical_gene_id("   ") is None
    assert g.canonical_gene_ids(["ENSG00000005955", "ENSG00000278311"]) == [
        "ENSG00000278311",
        "ENSG00000278311",
    ]


def test_canonical_gene_symbol_display_and_short_names():
    assert g.canonical_gene_symbol("ENSG00000141510") == "TP53"
    assert g.canonical_gene_symbol("ENSG00000005955") == "GGNBP2"  # retired id -> current
    assert g.canonical_gene_symbol("7157") == "TP53"  # Entrez/GeneID -> symbol
    assert g.canonical_gene_symbol("GNB2L1") == "RACK1"  # previous symbol / alias
    assert g.canonical_gene_symbol("TCEB2") == "ELOB"  # previous symbol / alias
    assert g.canonical_gene_symbols(["TP53", "GNB2L1", "NOT_A_REAL_GENE"]) == [
        "TP53",
        "RACK1",
        None,
    ]

    assert g.display_gene_name("GNB2L1") == "RACK1"
    assert g.display_gene_name("NOT_A_REAL_GENE") == "NOT_A_REAL_GENE"
    assert g.display_gene_name("NOT_A_REAL_GENE", fallback=False) is None
    assert g.display_gene_name("ENSG99999999999") is None
    assert g.short_gene_name("ENSG00000005955") == "GGNBP2"


def test_gene_display_helpers_are_top_level_exports():
    for name in (
        "canonical_gene_symbol",
        "canonical_gene_symbols",
        "display_gene_name",
        "entrez_gene_mappings",
        "gene_identifier_mapping_coverage",
        "gene_identifier_mapping_summary",
        "resolve_entrez_id",
        "short_gene_name",
    ):
        assert name in oncoref.__all__
        assert hasattr(oncoref, name)


def test_entrez_gene_mapping_helpers():
    mappings = g.entrez_gene_mappings()
    assert len(mappings) > 60_000
    assert {
        "entrez_id",
        "live_entrez_id",
        "canonical_ensembl_gene_id",
        "canonical_symbol",
        "mapping_method",
        "current_mapping_method",
    } <= set(mappings.columns)

    assert g.resolve_entrez_id("7157") == (
        "ENSG00000141510",
        "TP53",
        "entrez_dbxrefs",
    )
    assert g.resolve_entrez_id("not_numeric") is None
    assert oncoref.resolve_entrez_id("5728")[:2] == ("ENSG00000171862", "PTEN")


def test_ncbi_mitochondrial_gene_ids_use_explicit_symbol_translation():
    assert g.resolve_entrez_id("4512") == (
        "ENSG00000198804",
        "MT-CO1",
        "entrez_mitochondrial_symbol",
    )
    assert g.resolve_entrez_id("4550") == (
        "ENSG00000210082",
        "MT-RNR2",
        "entrez_mitochondrial_symbol",
    )


def test_gene_identifier_mapping_coverage_reports_shipped_mapping_boundaries():
    coverage = g.gene_identifier_mapping_coverage()
    expected = {
        "ensembl_gene_id",
        "symbol",
        "biotype",
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
    }
    assert expected <= set(coverage.columns)
    assert len(coverage) == len(g.canonical_gene_space())
    assert "ok" in set(coverage["mapping_status"])

    keyed = coverage.set_index("symbol")
    assert keyed.loc["TP53", "symbol_roundtrip"]
    assert keyed.loc["TP53", "has_entrez_id"]
    assert keyed.loc["RACK1", "n_symbol_aliases"] > 0
    assert keyed.loc["GGNBP2", "n_ensembl_aliases"] > 0
    assert (
        coverage["symbol_roundtrip_issue"].isin({"ok", "missing_symbol", "non_unique_symbol"}).all()
    )
    assert coverage["symbol_roundtrip_issue"].eq("non_unique_symbol").sum() > 0

    summary = g.gene_identifier_mapping_summary().iloc[0]
    assert summary["n_genes"] == len(coverage)
    assert summary["n_symbol_alias_rows"] >= 60_000
    assert summary["n_ensembl_alias_rows"] >= 6_000
    assert summary["n_entrez_mapping_rows"] > 60_000
    assert summary["n_with_entrez_ids"] > 40_000
    assert summary["n_without_symbol"] > 0
    assert summary["n_symbol_roundtrip_failed"] > 0
    assert summary["n_symbol_roundtrip_failed"] == summary["n_non_unique_symbols"]
    assert summary["n_symbol_not_resolvable"] == 0
    assert summary["n_symbol_resolves_to_other_gene"] == 0
    assert "ok" in summary["mapping_statuses"].split(";")
    assert "non_unique_symbol" in summary["symbol_roundtrip_issues"].split(";")


def test_symbol_synonym_resolution():
    syn = g.symbol_synonyms()
    assert len(syn) > 1000
    alias, official = next(iter(syn.items()))
    assert g.resolve_symbol(alias) == official
    assert g.resolve_symbol(alias.lower()) == official  # case-insensitive
    # an unknown / already-official symbol passes through
    assert g.resolve_symbol("NOT_A_REAL_GENE") == "NOT_A_REAL_GENE"


def test_symbol_synonym_resolution_preserves_exact_case_collisions():
    assert g.resolve_symbol("NA") == "XK"
    assert g.resolve_symbol("NaN") == "SCN11A"
    assert g.resolve_symbol("20-ALPHA-HSD") == "AKR1C1"
    assert g.resolve_symbol("20-alpha-HSD") == "HSD17B1"
    assert g.resolve_symbol("5PTASE") == "INPP5A"
    assert g.resolve_symbol("5PTase") == "INPP5B"

    # Other casing deterministically follows the exact uppercase snapshot row.
    assert g.resolve_symbol("20-AlPhA-HsD") == "AKR1C1"
    assert g.resolve_symbol("5PtAsE") == "INPP5A"
    assert g.canonical_gene_id("20-ALPHA-HSD") == "ENSG00000187134"
    assert g.canonical_gene_id("20-alpha-HSD") == "ENSG00000108786"
    assert g.canonical_gene_id("5PTASE") == "ENSG00000068383"
    assert g.canonical_gene_id("5PTase") == "ENSG00000204084"


def test_symbol_synonym_resolution_leaves_unanchored_case_collisions_ambiguous():
    assert g.resolve_symbol("H-plk") == "ZNF117"
    assert g.resolve_symbol("h-PLK") == "ERV3-1-ZNF117"
    assert g.resolve_symbol("H-PLK") == "H-PLK"


def test_every_exact_symbol_synonym_row_roundtrips():
    synonyms = get_data("ncbi-symbol-synonyms", copy=False)
    failures = [
        (str(row.alias), str(row.official_symbol), g.resolve_symbol(str(row.alias)))
        for row in synonyms.itertuples(index=False)
        if g.resolve_symbol(str(row.alias)) != str(row.official_symbol)
    ]
    assert not failures, failures[:5]


def test_canonical_gene_space_and_biotype():
    sp = g.canonical_gene_space()
    assert {"ensembl_gene_id", "symbol", "biotype", "seqname"} <= set(sp.columns)
    assert sp["ensembl_gene_id"].is_unique  # one canonical row per gene
    assert (sp["biotype"] == "protein_coding").sum() > 19_000  # the ~20k coding set
    assert sp["seqname"].isin({*(str(i) for i in range(1, 23)), "X", "Y", "MT"}).all()

    # biotype / coding lookups, resolving an old/alt id through the migration map first
    assert g.gene_biotype("ENSG00000141510") == "protein_coding"  # TP53
    assert g.is_protein_coding_gene("ENSG00000141510")
    assert not g.is_protein_coding_gene("ENSG00000251562")  # MALAT1 (lncRNA)
    assert g.gene_biotype("ENSG00000005955") == "protein_coding"  # old GGNBP2 id -> resolved
    # membership: a real gene is in the space; a non-Ensembl/garbage id is not
    assert g.is_canonical_gene("ENSG00000141510")
    assert not g.is_canonical_gene("ENSG99999999999")


def test_loaders_have_expected_columns():
    assert {"transcript_id", "ensembl_gene_id"} <= set(g.extra_transcript_mappings().columns)
    assert {"ensembl_gene_id", "n_members"} <= set(g.cdna_identical_groups().columns)
    assert "reason" in g.proteoform_collapse_overrides().columns
