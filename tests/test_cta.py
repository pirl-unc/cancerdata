# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import oncoref
from oncoref import cta


def test_set_sizes_reasonable():
    # Robust to curation refreshes: assert the sets are substantial and that
    # symbols and IDs agree in cardinality, rather than pinning exact counts
    # (which drift whenever the source databases or HPA version update).
    expressed = cta.cta_gene_names()
    assert len(expressed) == len(cta.cta_gene_ids())
    assert 150 < len(expressed) < len(cta.cta_unfiltered_gene_names())
    assert len(cta.cta_unfiltered_gene_ids()) > 250


def test_set_relationships():
    expressed = cta.cta_gene_names()
    filtered = cta.cta_filtered_gene_names()
    unfiltered = cta.cta_unfiltered_gene_names()
    assert expressed <= filtered <= unfiltered
    # never-expressed = filtered minus expressed
    assert cta.cta_never_expressed_gene_names() == filtered - expressed
    # excluded = unfiltered minus filtered (fail reproductive restriction)
    assert cta.cta_excluded_gene_names() == unfiltered - filtered


def test_uppercase_exports_are_compatibility_aliases():
    aliases = {
        "CTA_gene_names": "cta_gene_names",
        "CTA_gene_ids": "cta_gene_ids",
        "CTA_filtered_gene_names": "cta_filtered_gene_names",
        "CTA_filtered_gene_ids": "cta_filtered_gene_ids",
        "CTA_never_expressed_gene_names": "cta_never_expressed_gene_names",
        "CTA_never_expressed_gene_ids": "cta_never_expressed_gene_ids",
        "CTA_unfiltered_gene_names": "cta_unfiltered_gene_names",
        "CTA_unfiltered_gene_ids": "cta_unfiltered_gene_ids",
        "CTA_excluded_gene_names": "cta_excluded_gene_names",
        "CTA_excluded_gene_ids": "cta_excluded_gene_ids",
        "CTA_testis_restricted_gene_names": "cta_testis_restricted_gene_names",
        "CTA_testis_restricted_gene_ids": "cta_testis_restricted_gene_ids",
        "CTA_placental_restricted_gene_names": "cta_placental_restricted_gene_names",
        "CTA_placental_restricted_gene_ids": "cta_placental_restricted_gene_ids",
        "CTA_relaxed_reproductive_gene_names": "cta_relaxed_reproductive_gene_names",
        "CTA_relaxed_reproductive_gene_ids": "cta_relaxed_reproductive_gene_ids",
        "CTA_by_axes": "cta_by_axes",
    }
    for alias, canonical in aliases.items():
        assert getattr(cta, alias) is getattr(cta, canonical)
        assert getattr(oncoref, alias) is getattr(cta, canonical)


def test_canonical_ctas_present():
    expressed = cta.cta_gene_names()
    for g in ("MAGEA4", "MAGEA1", "CTAG1B", "PRAME"):
        assert g in expressed


def test_never_expressed_rescue_is_a_uniform_rule():
    # never_expressed CTAs with MODERATE confidence + STRICT reproductive RNA are
    # kept in the expressed set by a uniform rule (not a one-gene XAGE5 override).
    # XAGE5 is rescued, and so are its peers with the same signature.
    expressed = cta.cta_gene_ids()
    assert "ENSG00000171405" in expressed  # XAGE5
    assert "MAGEA2B" in cta.cta_gene_names()  # a same-signature peer, also kept
    # The rescue is exactly the rule, applied to every row.
    df = cta.cta_df()
    rescued = df[cta._never_expressed_rescue_mask(df)]
    never = rescued["never_expressed"].astype(str).str.lower() == "true"
    kept = set(rescued.loc[never, "Ensembl_Gene_ID"].astype(str).str.split(".").str[0])
    assert kept and kept <= cta.cta_gene_ids()


def test_non_cta_excluded_genes_dropped():
    # Histones / tubulins flagged out of the CTA universe entirely.
    df = cta.cta_df()
    unversioned = set(df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0])
    assert unversioned.isdisjoint(cta.NON_CTA_EXCLUDED_GENE_IDS)
    assert cta.NON_CTA_EXCLUDED_GENE_IDS.isdisjoint(cta.cta_unfiltered_gene_ids())


def test_no_histone_or_tubulin_survives_in_cta_universe():
    # The exclusion is a gene-family rule, not a hand-list: EVERY core histone
    # and alpha-tubulin candidate is dropped, so a sibling can't be left in (the
    # H1-6-vs-H2BC1 inconsistency the family rule fixed). Guards future drift.
    from oncoref.gene_families import gene_family_ids
    from oncoref.load_dataset import get_data

    raw = get_data("cancer-testis-antigens")
    uid = raw["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    histone_candidates = set(uid[uid.isin(gene_family_ids("histone"))])
    tubulin_candidates = set(uid[raw["Symbol"].str.match(r"^TUBA\d", na=False)])
    survivors = (histone_candidates | tubulin_candidates) & cta.cta_unfiltered_gene_ids()
    assert not survivors, f"housekeeping structural genes left in the CTA universe: {survivors}"


def test_cgb8_not_deny_listed():
    # #20: CGB8 was hardcoded out as "placental hCG-beta" while its hCG-beta
    # siblings (CGB1/2/3/5/7) flow through the normal HPA filter. CGB8 must be
    # curated by that same filter, not a one-gene deny-list. It passes the filter
    # (protein REPRODUCTIVE), so it lands in the filtered set like CGB2.
    assert "ENSG00000213030" not in cta.NON_CTA_EXCLUDED_GENE_IDS
    assert "CGB8" in cta.cta_unfiltered_gene_names()
    assert "CGB8" in cta.cta_filtered_gene_names()


def test_evidence_has_no_ms_columns():
    # MS-runtime columns stay in the target-selection layer, not oncoref.
    cols = set(cta.cta_evidence().columns)
    assert not any(c.startswith("ms_") for c in cols)
    # but the HPA-derived restriction columns are present
    for c in ("passes_filters", "never_expressed", "protein_restriction", "rna_restriction"):
        assert c in cols


def test_shipped_restriction_is_hpa_only_synthesis():
    # Every shipped restriction/confidence pair must equal the HPA-only synthesis
    # of its own row — i.e. no MS contribution leaked into the bundled table.
    df = cta.cta_evidence()
    for _, row in df.iterrows():
        tissue, conf = cta.synthesize_restriction(row)
        assert str(row["restriction"]) == tissue
        assert str(row["restriction_confidence"]) == conf


def test_synthesize_restriction_drops_ms():
    # The synthesis ignores MS columns entirely.
    row = {
        "protein_restriction": "TESTIS",
        "protein_reliability": "Enhanced",
        "rna_restriction": "TESTIS",
        "rna_restriction_level": "STRICT",
        "ms_restriction": "RECURRENT_HEALTHY",  # would lower MS-aware confidence
    }
    tissue, conf = cta.synthesize_restriction(row)
    assert tissue == "TESTIS"
    assert conf == "HIGH"  # protein(1.5)+rna-agree(1.5) over 2 sources = 1.5 >= 1.2


def test_reproductive_rna_does_not_agree_with_somatic_protein():
    row = {
        "protein_restriction": "SOMATIC",
        "protein_reliability": "Enhanced",
        "rna_restriction": "REPRODUCTIVE",
        "rna_restriction_level": "STRICT",
    }
    assert cta.synthesize_restriction(row) == ("SOMATIC", "LOW")


def test_reproductive_rna_agrees_with_reproductive_protein():
    for protein_restriction in ("TESTIS", "PLACENTAL", "REPRODUCTIVE"):
        row = {
            "protein_restriction": protein_restriction,
            "protein_reliability": "Enhanced",
            "rna_restriction": "REPRODUCTIVE",
            "rna_restriction_level": "STRICT",
        }
        assert cta.synthesize_restriction(row) == (protein_restriction, "HIGH")


def test_gene_id_to_name():
    m = cta.cta_gene_id_to_name()
    assert len(m) == len(cta.cta_gene_ids())
    assert all(not k.count(".") for k in m)  # unversioned keys


def test_id_set_helpers_match_name_set_relationships():
    assert cta.cta_never_expressed_gene_ids() == cta.cta_filtered_gene_ids() - cta.cta_gene_ids()
    assert (
        cta.cta_excluded_gene_ids() == cta.cta_unfiltered_gene_ids() - cta.cta_filtered_gene_ids()
    )


def test_axis_helpers_and_filters():
    testis = cta.cta_testis_restricted_gene_names()
    assert "CTAG1B" in testis
    assert testis == cta.cta_by_axes(restriction="TESTIS")
    assert testis <= cta.cta_filtered_gene_names()
    assert len(cta.cta_testis_restricted_gene_ids()) == len(testis)
    assert cta.cta_placental_restricted_gene_names()
    assert cta.cta_by_axes(ms_restriction="RECURRENT_HEALTHY") == set()
    all_restrictions = {"TESTIS", "PLACENTAL", "REPRODUCTIVE", "SOMATIC", "NO_DATA"}
    assert cta.cta_by_axes(restriction=all_restrictions) == cta.cta_filtered_gene_names()
    assert (
        cta.cta_by_axes(restriction=all_restrictions, filtered_only=False)
        == cta.cta_unfiltered_gene_names()
    )


def test_relaxed_reproductive_tier_is_opt_in():
    relaxed = cta.cta_relaxed_reproductive_gene_names()
    assert {"ERVW-1", "ERVFRD-1"} <= relaxed
    assert relaxed <= cta.cta_excluded_gene_names()
    assert not (relaxed & cta.cta_filtered_gene_names())
    assert cta.cta_relaxed_reproductive_gene_names(0.95) <= relaxed
    assert len(cta.cta_relaxed_reproductive_gene_ids()) == len(relaxed)


def test_cta_symbol_for_alias_resolves_common_names():
    assert cta.cta_symbol_for_alias("NY-ESO-1") == "CTAG1B"
    assert cta.cta_symbol_for_alias("ESO1") == "CTAG1B"
    assert cta.cta_symbol_for_alias("ny eso 1") == "CTAG1B"
    assert cta.cta_symbol_for_alias("CT12.2") == "XAGE2"
    assert cta.cta_symbol_for_alias("XAGE1C") == "XAGE1B"
    assert cta.cta_symbol_for_alias("MAGE4") == "MAGEA4"
    assert cta.cta_symbol_for_alias("MAGEA4") == "MAGEA4"
    assert cta.cta_symbol_for_alias("not-a-gene") is None


def test_cta_candidate_references_registry():
    # Top-of-funnel referenced candidate watchlist: every row carries an Ensembl
    # ID, a citation, and an HPA-restriction flag; none overlaps the curated set.
    cand = cta.cta_candidate_references()
    assert len(cand) >= 10
    required = {
        "Symbol",
        "Ensembl_Gene_ID",
        "candidate_source",
        "pmids",
        "hpa_testis_ntpm",
        "hpa_testis_restricted",
    }
    assert required <= set(cand.columns)
    assert cand["Ensembl_Gene_ID"].str.startswith("ENSG").all()
    assert cand["pmids"].astype(str).str.strip().ne("").all()  # every candidate cited
    # Watchlist genes are genuinely ABSENT from the table — not merely filtered
    # out of the passing view. Assert against every row's id (passing or not), so
    # a gene already sitting non-passing in the table can't sneak in here.
    table_ids = set(cta.cta_df()["Ensembl_Gene_ID"].astype(str).str.split(".").str[0])
    assert not (set(cand["Ensembl_Gene_ID"]) & table_ids)


def test_clinical_cta_target_tier_keeps_strict_default_conservative():
    clinical = {
        "CTAG2",
        "MAGEA11",
        "CT45A1",
        "CT45A3",
        "CT45A5",
        "MAGEC3",
        "PAGE4",
        "CSAG1",
    }
    leaky = {"CTAG2", "MAGEA11", "CT45A1", "CT45A3", "CT45A5"}

    assert clinical <= cta.cta_clinical_target_gene_names()
    assert leaky <= cta.cta_excluded_clinical_target_gene_names()
    assert leaky <= cta.cta_excluded_gene_names()
    assert not (leaky & cta.cta_filtered_gene_names())
    assert not ({"MAGEC3", "PAGE4", "CSAG1"} & cta.cta_unfiltered_gene_names())

    evidence = cta.cta_clinical_target_evidence().set_index("Symbol")
    assert set(evidence.loc[sorted(leaky), "evidence_tier"]) == {"excluded"}
    assert evidence.loc["CTAG2", "exclusion_driver_tissue"] == "heart muscle"
    assert evidence.loc["CT45A3", "exclusion_driver_tissue"] == "hippocampal formation"
    assert evidence.loc["CT45A5", "exclusion_driver_tissue"] == "basal ganglia"
    assert float(evidence.loc["CTAG2", "exclusion_driver_ntpm"]) == 5.1

    assert evidence.loc["MAGEC3", "evidence_tier"] == "candidate"
    assert evidence.loc["PAGE4", "evidence_tier"] == "candidate"
    assert evidence.loc["CSAG1", "evidence_tier"] == "candidate"
    assert evidence.loc["CSAG1", "candidate_source"] == "literature_cta"
    assert "PMID:12039054" in evidence.loc["CSAG1", "pmids"]
    assert "ENSG00000198930" in cta.cta_clinical_target_gene_ids()
    assert "ENSG00000198930" not in cta.cta_excluded_clinical_target_gene_ids()


def test_cta_specificity_audit_surfaces_demotions_and_candidate_only_rows():
    audit = cta.cta_specificity_audit().set_index("Symbol")
    expected = {
        "CTAG2",
        "MAGEA11",
        "CT45A1",
        "CT45A3",
        "CT45A5",
        "MAGEC3",
        "PAGE4",
        "CSAG1",
    }
    assert expected <= set(audit.index)

    assert audit.loc["CTAG2", "specificity_status"] == "excluded_somatic_leak"
    assert audit.loc["CTAG2", "specificity_action"] == "exclude_default_keep_clinical"
    assert audit.loc["CTAG2", "rna_max_somatic_tissue"] == "heart muscle"
    assert float(audit.loc["CTAG2", "rna_max_somatic_ntpm"]) == 5.1
    assert audit.loc["CTAG2", "in_cta_table"]

    assert audit.loc["PAGE4", "specificity_status"] == "candidate_weak_specificity"
    assert audit.loc["PAGE4", "candidate_source"] == "literature_cta"
    assert not audit.loc["PAGE4", "in_cta_table"]
    assert audit.loc["CSAG1", "specificity_status"] == "candidate_pending_hpa_audit"
    assert "PMID:12039054" in audit.loc["CSAG1", "pmids"]


def test_cta_evidence_includes_specificity_status_for_audited_and_default_rows():
    evidence = cta.cta_evidence().set_index("Symbol")

    assert evidence.loc["CTAG2", "specificity_status"] == "excluded_somatic_leak"
    assert evidence.loc["CTAG2", "specificity_action"] == "exclude_default_keep_clinical"
    assert "heart" in evidence.loc["CTAG2", "specificity_rationale"]
    assert evidence.loc["CTAG1B", "specificity_status"] == "canonical_default"
    assert evidence.loc["CTAG1B", "specificity_action"] == "include_default"
    assert "CTAG2" not in cta.cta_gene_names()


def test_default_cta_sets_honor_specificity_audit_actions():
    evidence = cta.cta_evidence()
    default = evidence["specificity_action"].astype(str) == "include_default"
    filtered = default | evidence["specificity_status"].astype(str).eq("canonical_low_expression")

    assert cta.cta_gene_names() == set(evidence.loc[default, "Symbol"].astype(str))
    assert cta.cta_gene_ids() == set(
        evidence.loc[default, "Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    )
    assert cta.cta_filtered_gene_names() == set(evidence.loc[filtered, "Symbol"].astype(str))
    assert cta.cta_filtered_gene_ids() == set(
        evidence.loc[filtered, "Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    )

    audited_demotions = set(
        evidence.loc[
            evidence["specificity_action"].astype(str) == "exclude_default_keep_clinical",
            "Symbol",
        ].astype(str)
    )
    assert audited_demotions
    assert not (audited_demotions & cta.cta_filtered_gene_names())
    assert audited_demotions <= cta.cta_excluded_clinical_target_gene_names()


def test_clinical_cta_helpers_are_top_level_exports():
    for name in (
        "cta_clinical_target_references",
        "cta_clinical_target_evidence",
        "cta_clinical_target_gene_names",
        "cta_clinical_target_gene_ids",
        "cta_excluded_clinical_target_gene_names",
        "cta_excluded_clinical_target_gene_ids",
        "cta_specificity_audit_references",
        "cta_specificity_audit",
    ):
        assert getattr(oncoref, name) is getattr(cta, name)
