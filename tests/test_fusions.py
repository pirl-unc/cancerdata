# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Driver fusions per cancer type + reverse lookups (#27, O4)."""

import pandas as pd
import pytest

from oncoref import (
    cancer_driver_spectrum,
    cancer_fusion_citation_audit,
    cancer_fusions,
    cancer_fusions_df,
    cancer_type_registry,
    cancer_types_with_fusion,
    fusion_partners,
    fusion_status,
    fusions,
    protein_family,
)


def test_ifs_driver_model_is_heterogeneous_and_diagnosis_independent():
    assert fusion_status("SARC_IFS") == {"status": "heterogeneous", "driver": ""}
    spectrum = cancer_driver_spectrum("SARC_IFS")
    assert {
        "ETV6-NTRK3",
        "EML4-NTRK3",
        "BRAF internal deletion",
        "molecularly unresolved",
    } == set(spectrum["driver_event"])

    ifs = cancer_fusions("SARC_IFS")
    assert set(zip(ifs["gene_5prime"], ifs["gene_3prime"])) == {
        ("ETV6", "NTRK3"),
        ("EML4", "NTRK3"),
    }
    assert not ifs["is_defining"].astype(str).str.lower().eq("true").any()
    assert "SARC_IFS" not in cancer_types_with_fusion("ETV6-NTRK3", defining_only=True)


def test_cmn_and_ntrk_spindle_are_distinct_entities_with_nondefining_overlap():
    cmn_events = set(cancer_driver_spectrum("CMN")["driver_event"])
    assert {"EGFR kinase-domain ITD", "ETV6-NTRK3", "LMNA-NTRK1"} <= cmn_events
    assert "SARC_NTRK_SPINDLE" not in set(cancer_driver_spectrum("CMN")["cancer_code"])
    assert "CMN" not in cancer_types_with_fusion("ETV6-NTRK3", defining_only=True)


def test_fusions_for_subtype():
    # The headline example: the alveolar-RMS subtype's PAX-FOXO1 fusion.
    arms = cancer_fusions("SARC_RMS_ARMS")
    pairs = set(zip(arms["gene_5prime"], arms["gene_3prime"]))
    assert ("PAX3", "FOXO1") in pairs
    assert ("PAX7", "FOXO1") in pairs


def test_astb_fusion_side_tables_match_registry_driver():
    astb = cancer_fusions("ASTB", defining_only=True)
    pairs = set(zip(astb["gene_5prime"], astb["gene_3prime"]))
    assert pairs == {("MN1", "BEND2")}
    assert cancer_types_with_fusion("MN1-BEND2", defining_only=True) == ["ASTB"]
    assert fusions.fusion_surrogate_genes_for_cancer("ASTB") == ["BEND2"]


def test_defining_only_filter():
    ews = cancer_fusions("SARC_EWS", defining_only=True)
    assert (ews["is_defining"].astype(str).str.lower() == "true").all()
    assert "EWSR1" in set(ews["gene_5prime"])


def test_reverse_lookup_by_fusion():
    assert cancer_types_with_fusion("EWSR1-FLI1") == ["SARC_EWS"]
    # `::` separator also accepted.
    assert cancer_types_with_fusion("EWSR1::FLI1") == ["SARC_EWS"]


def test_reverse_lookup_by_partner_family():
    fet = cancer_types_with_fusion(partner_family="FET")
    assert "SARC_EWS" in fet and "SARC_DSRCT" in fet


def test_reverse_lookup_requires_exactly_one_arg():
    with pytest.raises(ValueError, match="exactly one"):
        cancer_types_with_fusion("EWSR1-FLI1", partner="EWSR1")
    with pytest.raises(ValueError, match="exactly one"):
        cancer_types_with_fusion()


def test_fusion_partners_promiscuous():
    partners = fusion_partners("EWSR1")
    assert {"FLI1", "ERG"} <= partners


def test_fusion_partners_bad_side():
    with pytest.raises(ValueError, match="side must be"):
        fusion_partners("EWSR1", side="middle")


def test_protein_family():
    assert protein_family("PAX3") == "PAX"
    assert protein_family("FLI1") == "ETS"
    assert protein_family("NOT_A_GENE") is None


def test_nan_query_does_not_match_fusion_negative_rows():
    # The "(none)" fusion-negative rows have NaN gene names; a 'NAN'/'nan' query
    # must NOT match them (regression: .astype(str) turned NaN into "NAN").
    assert cancer_types_with_fusion(partner="NAN") == []
    assert cancer_types_with_fusion(partner="nan") == []
    assert cancer_types_with_fusion("NAN-NAN") == []
    assert fusion_partners("NAN") == set()
    assert protein_family("NAN") is None


def test_every_fusion_code_is_a_registry_code():
    # A fusion row must never reference a cancer code that doesn't exist.
    fusion_codes = set(cancer_fusions_df()["cancer_code"])
    codes = set(cancer_type_registry()["code"])
    missing = sorted(fusion_codes - codes)
    assert not missing, f"fusion cancer_codes not in the registry: {missing}"


def test_every_fusion_defined_registry_entity_has_defining_detail_rows():
    registry = cancer_type_registry()
    expected = set(registry.loc[registry["fusion_driven"] == "defining", "code"])
    details = cancer_fusions(defining_only=True)
    represented = set(details["cancer_code"])

    assert expected <= represented


def test_new_who_fusion_entities_own_their_detail_rows():
    expected_pairs = {
        "SARC_EWSR1_NONETS": {
            ("EWSR1", "NFATC2"),
            ("EWSR1", "PATZ1"),
            ("EWSR1", "SMARCA5"),
        },
        "SARC_SEF": {
            ("EWSR1", "CREB3L1"),
            ("EWSR1", "CREB3L2"),
            ("EWSR1", "CREB3L3"),
        },
        "SARC_NTRK_SPINDLE": {
            ("TPM3", "NTRK1"),
            ("LMNA", "NTRK1"),
            ("STRN", "NTRK2"),
            ("SQSTM1", "NTRK3"),
        },
        "SARC_EIMS": {("RANBP2", "ALK")},
    }
    for code, pairs in expected_pairs.items():
        rows = cancer_fusions(code, defining_only=True)
        assert set(zip(rows["gene_5prime"], rows["gene_3prime"])) == pairs

    malignant_pmt = cancer_fusions("SARC_PMT_MALIGNANT")
    assert set(zip(malignant_pmt["gene_5prime"], malignant_pmt["gene_3prime"])) == {
        ("FN1", "FGFR1")
    }
    assert not malignant_pmt["is_defining"].astype(str).str.lower().eq("true").any()


def test_every_positive_fusion_or_alteration_has_a_citation():
    fusions_df = cancer_fusions_df().fillna("")
    positive_rows = fusions_df[fusions_df["fusion_family"] != "(none)"]
    assert positive_rows["pmid"].str.fullmatch(r"PMID:\d+").all()


def test_fusion_citation_audit_covers_every_cited_row():
    key = ["cancer_code", "fusion_family", "gene_5prime", "gene_3prime", "pmid"]
    fusions_df = cancer_fusions_df().fillna("")
    cited = fusions_df[fusions_df["pmid"] != ""][key].sort_values(key).reset_index(drop=True)

    audit = cancer_fusion_citation_audit().fillna("")
    audited = audit[key].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(audited, cited)

    assert audit["pubmed_title"].str.strip().ne("").all()
    assert audit["publication_year"].between(1900, 2100).all()
    assert audit["supports_fusion_or_alteration"].all()
    assert audit["supports_disease_context"].all()
    reviewed_on = pd.to_datetime(audit["reviewed_on"], format="%Y-%m-%d", errors="raise")
    assert reviewed_on.notna().all()


def test_fusion_citation_audit_returns_a_defensive_copy():
    first = cancer_fusion_citation_audit()
    first.loc[0, "pubmed_title"] = "changed"
    assert cancer_fusion_citation_audit().loc[0, "pubmed_title"] != "changed"


def test_known_wrong_fusion_pmids_are_replaced():
    """Issue #160 examples: real PMIDs that resolved to unrelated papers."""

    fusions_df = cancer_fusions_df()
    known_bad_pmids = {
        "PMID:7951326",
        "PMID:7954420",
        "PMID:9590769",
        "PMID:21885844",
        "PMID:9537325",
        "PMID:16462738",
        "PMID:9192848",
        "PMID:11427703",
        "PMID:8316832",
        "PMID:1565502",
        "PMID:6304885",
        "PMID:3929080",
        "PMID:6262918",
        "PMID:9596662",
        "PMID:10866930",
    }
    observed_pmids = set(fusions_df["pmid"].dropna())
    assert known_bad_pmids.isdisjoint(observed_pmids)

    expected_pmids = {
        ("SARC_SYN", "SS18-SSX", "SSX1"): "PMID:7951320",
        ("SARC_SYN", "SS18-SSX", "SSX2"): "PMID:7951320",
        ("SARC_SYN", "SS18-SSX", "SSX4"): "PMID:11368913",
        ("SARC_DSRCT", "EWSR1-WT1", "WT1"): "PMID:7862627",
        ("SARC_IFS", "ETV6-NTRK3", "NTRK3"): "PMID:9462753",
        ("SARC_EHE", "WWTR1-CAMTA1", "CAMTA1"): "PMID:21885404",
        ("SARC_EMC", "EWSR1-NR4A3", "NR4A3"): "PMID:18855877",
        ("SARC_IMT", "RANBP2-ALK", "ALK"): "PMID:24034896",
        ("SARC_DFSP", "COL1A1-PDGFB", "PDGFB"): "PMID:31949795",
        ("SARC_ESS_LG", "JAZF1-SUZ12", "SUZ12"): "PMID:16049311",
        ("LAML", "CBFB-MYH11", "MYH11"): "PMID:8142642",
        ("LAML", "DEK-NUP214", "NUP214"): "PMID:32526729",
        ("CML", "BCR-ABL1", "ABL1"): "PMID:40360311",
        ("FL", "BCL2-IGH", "BCL2"): "PMID:18684042",
        ("BL", "MYC-IGH", "MYC"): "PMID:23673335",
        ("MM", "CCND1-IGH", "CCND1"): "PMID:35982976",
        ("THCA", "PAX8-PPARG", "PPARG"): "PMID:25069464",
    }
    for (cancer_code, fusion_family, gene_3prime), expected_pmid in expected_pmids.items():
        row = fusions_df[
            (fusions_df["cancer_code"] == cancer_code)
            & (fusions_df["fusion_family"] == fusion_family)
            & (fusions_df["gene_3prime"] == gene_3prime)
        ]
        assert len(row) == 1
        assert row.iloc[0]["pmid"] == expected_pmid


def test_issue_160_claim_corrections():
    fusions_df = cancer_fusions_df().fillna("")
    imt_pairs = set(
        zip(
            fusions_df.loc[fusions_df["cancer_code"] == "SARC_IMT", "gene_5prime"],
            fusions_df.loc[fusions_df["cancer_code"] == "SARC_IMT", "gene_3prime"],
        )
    )
    assert ("CARS1", "ALK") in imt_pairs
    assert ("CARS1", "ROS1") not in imt_pairs

    rms_pairs = set(
        zip(
            fusions_df.loc[fusions_df["cancer_code"] == "SARC_RMS_ARMS", "gene_5prime"],
            fusions_df.loc[fusions_df["cancer_code"] == "SARC_RMS_ARMS", "gene_3prime"],
        )
    )
    assert ("PAX3", "FOXO4") not in rms_pairs


def test_registry_fusion_drivers_are_present_in_fusion_table():
    fusions_df = cancer_fusions_df().fillna("")
    registry = cancer_type_registry().fillna("")

    pairs_by_code = {}
    for row in fusions_df.itertuples():
        if row.gene_5prime and row.gene_3prime:
            pairs_by_code.setdefault(row.cancer_code, set()).add(
                f"{row.gene_5prime}-{row.gene_3prime}"
            )

    missing = []
    for row in registry.itertuples():
        # Some ontology-only entities declare a driver before a detailed fusion
        # row exists. Compare only codes represented in both tables.
        if row.code not in pairs_by_code:
            continue
        declared = {value.strip() for value in row.fusion_driver.split(";") if value.strip()}
        for fusion in sorted(declared - pairs_by_code.get(row.code, set())):
            missing.append((row.code, fusion))

    assert not missing, f"registry fusion drivers missing from cancer-fusions.csv: {missing}"
