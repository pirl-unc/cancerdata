# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Normalization-reference gene families (#35, R-norm)."""

import pandas as pd
import pytest

import oncoref
from oncoref import gene_families as gf


def test_families_load_with_ids():
    for name in gf.gene_families():
        ids = gf.gene_family_ids(name)
        assert ids, f"{name} family is empty"
        assert all(i.startswith("ENSG") and "." not in i for i in ids)


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown gene family"):
        gf.gene_family("not_a_family")
    with pytest.raises(ValueError, match="unknown gene family"):
        gf.gene_family_ids("not_a_family")


def test_technical_rna_is_union_of_its_families():
    union = set()
    for fam in ("mitochondrial", "numt_pseudogene", "rrna", "nuclear_retained_lncrna"):
        union |= gf.gene_family_ids(fam)
    assert union <= gf.technical_rna_gene_ids()
    assert gf.technical_rna_gene_ids() == gf.clean_tpm_other_technical_gene_ids()
    assert gf.clean_tpm_protocol_sensitive_gene_ids() <= gf.technical_rna_gene_ids()
    assert gf.technical_rna_gene_ids()  # non-empty


def test_clean_tpm_compartment_ids_follow_censored_table_categories():
    censored = gf.clean_tpm_censored_gene_ids()
    ribosomal = gf.clean_tpm_ribosomal_gene_ids()
    other_technical = gf.clean_tpm_other_technical_gene_ids()

    assert len(ribosomal) == 1767
    assert len(other_technical) >= 6000
    assert ribosomal | other_technical == censored
    assert ribosomal.isdisjoint(other_technical)

    # RPL10AP1 is a ribosomal-protein pseudogene whose public censored-table category
    # is ribosomal_protein, so it belongs to the 16% budget, not the 9% remainder.
    assert "ENSG00000244691" in ribosomal
    assert "ENSG00000244691" not in other_technical


def test_clean_tpm_censors_protocol_sensitive_structural_rnas_but_not_mirna():
    structural = gf.clean_tpm_protocol_sensitive_gene_ids()
    other_technical = gf.clean_tpm_other_technical_gene_ids()

    assert structural <= other_technical
    assert {
        "ENSG00000276168",  # RN7SL1, misc_RNA
        "ENSG00000283293",  # RN7SK, snRNA
        "ENSG00000263934",  # SNORD3A, snoRNA
        "ENSG00000277027",  # RMRP, ribozyme
        "ENSG00000277209",  # RPPH1, ribozyme
    } <= structural
    assert "ENSG00000199017" not in structural  # MIR1-1 remains outside this class.


def test_clean_tpm_ribosomal_pseudogene_audit_ids_in_both_references():
    audited_ids = {
        "ENSG00000292328",  # RPL14P5
        "ENSG00000280437",  # RPL23AP53
        "ENSG00000293173",  # RPL31P11
        "ENSG00000283120",  # RPL11P3
        "ENSG00000283753",  # RPL21P121
        "ENSG00000283660",  # RPL23AP84
        "ENSG00000284212",  # RPL7AP65
        "ENSG00000242150",  # RPS10P7
        "ENSG00000282862",  # RPS21P1
        "ENSG00000282942",  # RPS26P3
        "ENSG00000284153",  # RPS4XP17
        "ENSG00000282839",  # RPS4XP18
    }

    assert audited_ids <= gf.clean_tpm_ribosomal_gene_ids()
    assert audited_ids <= gf.gene_family_ids("ribosomal_protein_pseudogene")


def test_clean_tpm_censoring_is_cta_safe():
    from oncoref import cta

    rpl10l = "ENSG00000165496"
    cta_universe = {str(g).split(".")[0] for g in cta.cta_unfiltered_gene_ids()}
    censored = gf.clean_tpm_censored_gene_ids()

    assert rpl10l in gf.gene_family_ids("ribosomal_protein")
    assert rpl10l in cta_universe
    assert rpl10l not in gf.clean_tpm_ribosomal_gene_ids()
    assert rpl10l not in gf.clean_tpm_other_technical_gene_ids()
    assert rpl10l not in censored
    assert cta_universe.isdisjoint(censored)


def test_housekeeping_panel():
    df = gf.housekeeping_genes()
    assert {"Symbol", "Ensembl_Gene_ID"} <= set(df.columns)
    assert "ACTB" in set(df["Symbol"])
    assert gf.housekeeping_gene_ids()


def test_clean_tpm_biological_housekeeping_panel_contract():
    from oncoref.gene_ids import is_protein_coding_gene

    full = gf.clean_tpm_biological_housekeeping_genes(primary_only=False)
    primary = gf.clean_tpm_biological_housekeeping_genes()

    assert len(full) == 47
    assert len(primary) == 30
    assert set(primary["Ensembl_Gene_ID"]) < set(full["Ensembl_Gene_ID"])
    assert {
        "Symbol",
        "Ensembl_Gene_ID",
        "hpa_tissue_count",
        "min_tpm",
        "mean_tpm",
        "cv",
        "max_tpm",
        "clean_tpm_component",
        "primary_panel",
        "selection_note",
    } <= set(full.columns)

    assert set(full["clean_tpm_component"]) == {"biological"}
    assert full["hpa_tissue_count"].eq(50).all()
    assert full["min_tpm"].ge(100).all()
    assert full["cv"].lt(0.5).all()
    assert full["Ensembl_Gene_ID"].map(is_protein_coding_gene).all()

    censored = gf.clean_tpm_censored_gene_ids()
    assert gf.clean_tpm_biological_housekeeping_gene_ids().isdisjoint(censored)
    assert gf.clean_tpm_biological_housekeeping_gene_ids(primary_only=False).isdisjoint(censored)


def test_clean_tpm_housekeeping_helpers_are_top_level_exports():
    for name in (
        "clean_tpm_biological_housekeeping_gene_ids",
        "clean_tpm_biological_housekeeping_genes",
        "clean_tpm_ribosomal_gene_ids",
        "clean_tpm_other_technical_gene_ids",
    ):
        assert hasattr(oncoref, name)
        assert name in oncoref.__all__


def test_clean_tpm_housekeeping_primary_excludes_legacy_ribosomal_references():
    primary_symbols = set(gf.clean_tpm_biological_housekeeping_genes()["Symbol"])
    full_symbols = set(gf.clean_tpm_biological_housekeeping_genes(primary_only=False)["Symbol"])

    assert {"SUMO2", "EEF1A1", "COX7A2"} <= primary_symbols
    assert "PPIA" in full_symbols and "PPIA" not in primary_symbols
    assert {
        "RPLP0",
        "RPL13A",
        "RPS18",
        "RPS13",
        "RPL19",
        "RPL27",
        "RPS27",
    }.isdisjoint(primary_symbols)


def test_legacy_qpcr_housekeeping_aliases_preserve_historical_panel():
    assert gf.legacy_qpcr_housekeeping_genes().equals(gf.housekeeping_genes())
    assert gf.legacy_qpcr_housekeeping_gene_ids() == gf.housekeeping_gene_ids()
    assert "ACTB" in set(gf.legacy_qpcr_housekeeping_genes()["Symbol"])


def test_censored_references():
    assert gf.clean_tpm_censored_gene_ids()  # the clean_tpm_v4 censored set
    table = gf.clean_tpm_censored_genes()
    assert {
        "Ensembl_Gene_ID",
        "Symbol",
        "category",
        "reference_tpm",
        "reference_source",
        "reference_profile_version",
    } == set(table.columns)
    assert set(table["reference_source"]) == {gf.CLEAN_TPM_CENSORED_REFERENCE_PROFILE_SOURCE}
    assert set(table["reference_profile_version"]) == {
        gf.CLEAN_TPM_CENSORED_REFERENCE_PROFILE_VERSION
    }
    assert table["reference_tpm"].ge(0).all()
    assert set(gf.clean_tpm_censored_reference_tpm()) == set(table["Ensembl_Gene_ID"])
    ref = gf.censored_gene_reference_tpm()
    assert ref and all(isinstance(v, float) and v >= 0 for v in ref.values())


def _hpa_rows(gene_id: str, symbol: str, values: list[float]) -> list[dict[str, object]]:
    return [
        {"Gene": gene_id, "Gene name": symbol, "Tissue": f"tissue_{i}", "nTPM": value}
        for i, value in enumerate(values)
    ]


def test_hpa_housekeeping_candidates_filter_to_biological_protein_coding(monkeypatch):
    monkeypatch.setattr(
        gf,
        "clean_tpm_censored_gene_ids",
        lambda include_ribosomal_proteins=True: frozenset({"ENSG_CENS"}),
    )
    hpa_rna = pd.DataFrame(
        _hpa_rows("ENSG_KEEP.5", "KEEP", [100.0, 120.0, 140.0])
        + _hpa_rows("ENSG_LOW", "LOW", [90.0, 120.0, 140.0])
        + _hpa_rows("ENSG_SPARSE", "SPARSE", [500.0])
        + _hpa_rows("ENSG_NAN", "NAN", [500.0, 520.0, None])
        + _hpa_rows("ENSG_CENS", "CENS", [200.0, 210.0, 220.0])
        + _hpa_rows("ENSG_NCRNA", "NCRNA", [200.0, 210.0, 220.0])
    )
    gene_space = pd.DataFrame(
        {
            "ensembl_gene_id": [
                "ENSG_KEEP",
                "ENSG_LOW",
                "ENSG_SPARSE",
                "ENSG_NAN",
                "ENSG_CENS",
                "ENSG_NCRNA",
            ],
            "biotype": [
                "protein_coding",
                "protein_coding",
                "protein_coding",
                "protein_coding",
                "protein_coding",
                "lncRNA",
            ],
        }
    )

    out = gf.hpa_housekeeping_candidates(hpa_rna, gene_space=gene_space)

    assert out["Symbol"].tolist() == ["KEEP"]
    row = out.iloc[0]
    assert row["Ensembl_Gene_ID"] == "ENSG_KEEP"
    assert row["min_ntpm"] == pytest.approx(100.0)
    assert row["mean_ntpm"] == pytest.approx(120.0)
    assert row["max_ntpm"] == pytest.approx(140.0)
    assert row["n_tissues"] == 3
    assert row["n_tissues_expected"] == 3
    # Population CV, matching the original HPA-wide derivation.
    assert row["cv"] == pytest.approx((((20.0**2 + 0.0 + 20.0**2) / 3) ** 0.5) / 120.0)


def test_recommended_hpa_housekeeping_panel_applies_review_policy(monkeypatch):
    monkeypatch.setattr(
        gf,
        "clean_tpm_censored_gene_ids",
        lambda include_ribosomal_proteins=True: frozenset(),
    )
    hpa_rna = pd.DataFrame(
        _hpa_rows("ENSG00000096384", "HSP90AB1", [210.0] + [500.0] * 8 + [700.0])
        # PPIA passes the numeric rule, but is a deliberate biological holdout.
        + _hpa_rows("ENSG00000196262", "PPIA", [356.0] + [900.0] * 8 + [1200.0])
        # HSP90AA1 passes, but is redundant with HSP90AB1 in the first-pass panel.
        + _hpa_rows("ENSG00000080824", "HSP90AA1", [186.0] + [500.0] * 8 + [700.0])
        # EEF1A1 exceeds the preferred range but is a protected high-abundance exception.
        + _hpa_rows("ENSG00000156508", "EEF1A1", [1624.0] + [5847.0] * 8 + [11871.0])
    )
    gene_space = pd.DataFrame(
        {
            "ensembl_gene_id": [
                "ENSG00000096384",
                "ENSG00000196262",
                "ENSG00000080824",
                "ENSG00000156508",
            ],
            "biotype": ["protein_coding"] * 4,
        }
    )

    out = gf.recommended_hpa_housekeeping_panel(hpa_rna, gene_space=gene_space)

    assert set(out["Symbol"]) == {"HSP90AB1", "EEF1A1"}
    assert set(out["Symbol"]).isdisjoint({"PPIA", "HSP90AA1"})
    eef = out.set_index("Symbol").loc["EEF1A1"]
    assert eef["max_min_ratio"] > gf.HPA_HOUSEKEEPING_MAX_MIN_RATIO
    assert eef["selection_reason"] == "high_abundance_literature_exception"
