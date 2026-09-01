# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math

import pandas as pd

from oncoref import apd1, ici


def test_regimens_and_table():
    assert ici.ici_regimens() == ("PD-1", "PD-L1", "PD-1+CTLA-4")
    df = ici.cancer_ici_response_df()
    assert {"cancer_code", "regimen", "orr_pct"} <= set(df.columns)
    # all three regimens are actually present (not just PD-1)
    assert set(df["regimen"]) == {"PD-1", "PD-L1", "PD-1+CTLA-4"}
    assert (df["regimen"] == "PD-L1").sum() >= 10  # anti-PD-L1 is well-represented


def test_ici_anchor_table_exposes_evidence_schema():
    df = ici.cancer_ici_response_df()
    expected = {
        "response_metric",
        "response_unit",
        "response_ci_low",
        "response_ci_high",
        "response_ci_basis",
        "response_ci_low_status",
        "response_ci_high_status",
        "response_value_status",
        "response_numerator",
        "response_denominator",
        "source_estimate_id",
        "source_locator",
        "source_locator_status",
        "source_endpoint_label",
        "source_population_label",
        "source_n",
        "source_verified",
        "value_basis",
        "source_anchor",
        "endpoint_population",
        "therapy_regimen_class",
        "evidence_type",
        "histology_match",
        "is_direct_cancer_code_evidence",
        "evidence_source_code",
        "source_scope",
        "missing_reason",
    }
    assert expected <= set(df.columns)

    crc = df[(df["cancer_code"] == "CRC_MSI") & (df["regimen"] == "PD-1")].iloc[0]
    assert crc["response_metric"] == "ORR"
    assert crc["response_numerator"] == 67
    assert crc["response_denominator"] == 153
    assert crc["response_ci_low"] == 35.8
    assert crc["response_ci_high"] == 52.0
    assert crc["response_ci_basis"] == "reported"
    assert crc["response_ci_low_status"] == "numeric"
    assert crc["response_ci_high_status"] == "numeric"
    assert crc["response_value_status"] == "numeric"
    assert crc["source_estimate_id"].startswith("ICI-")
    assert crc["source_locator"] == "Table 2"
    assert crc["source_locator_status"] == "verified"
    assert crc["therapy_regimen_class"] == "anti_pd1_monotherapy"
    assert crc["evidence_type"] == "direct_reported"
    assert crc["histology_match"] == "direct"
    assert bool(crc["is_direct_cancer_code_evidence"]) is True
    assert crc["source_scope"] == "aggregate_source"
    assert crc["source_anchor"] == "PMID:33264544"

    coad = df[(df["cancer_code"] == "COAD") & (df["regimen"] == "PD-1")].iloc[0]
    assert coad["evidence_type"] == "derived_blend"
    assert coad["histology_match"] == "derived"
    assert bool(coad["is_direct_cancer_code_evidence"]) is False
    assert coad["source_scope"] == "derived_blend"
    assert pd.isna(coad["response_denominator"])
    assert pd.isna(coad["source_anchor"])


def test_ici_estimates_expose_structured_source_and_ci_provenance():
    df = ici.cancer_ici_response_estimates_df()
    expected = {
        "estimate_id",
        "source_locator",
        "source_locator_status",
        "source_endpoint_label",
        "source_population_label",
        "value_status",
        "ci_low_status",
        "ci_high_status",
        "ci_basis",
    }
    assert expected <= set(df.columns)
    assert df["estimate_id"].is_unique
    assert df["estimate_id"].str.match(r"^ICI-[0-9a-f]{10}-[0-9]{2}$").all()
    assert set(df["source_locator_status"]) == {
        "citation_only",
        "located_unverified",
        "not_applicable",
        "not_verified",
        "source_section",
        "verified",
    }
    assert set(df["value_status"]) <= {
        "not_estimable",
        "not_reached",
        "not_reported",
        "not_verified",
        "numeric",
    }
    assert set(df["ci_low_status"]) <= {
        "not_applicable",
        "not_reported",
        "not_verified",
        "numeric",
        "source_unavailable",
    }
    assert set(df["ci_high_status"]) <= {
        "NE",
        "NR",
        "not_applicable",
        "not_reported",
        "not_verified",
        "numeric",
        "source_unavailable",
    }
    assert set(df["ci_basis"]) <= {
        "computed_clopper_pearson",
        "computed_wilson",
        "not_applicable",
        "not_reported",
        "not_verified",
        "reported",
        "source_unavailable",
    }
    provenance_columns = [
        "source_locator_status",
        "value_status",
        "ci_low_status",
        "ci_high_status",
        "ci_basis",
    ]
    assert not df[provenance_columns].eq("not_extracted").any().any()

    adcc_os = df[
        (df["cancer_code"] == "ADCC") & (df["regimen"] == "PD-1+CTLA-4") & (df["metric"] == "OS")
    ].iloc[0]
    assert adcc_os["ci_low_status"] == "numeric"
    assert adcc_os["ci_high_status"] == "NR"
    assert adcc_os["ci_basis"] == "reported"

    bcc_pfs = df[
        (df["cancer_code"] == "BCC")
        & (df["regimen"] == "PD-1")
        & (df["metric"] == "PFS")
        & (df["ci_high_status"] == "NE")
    ].iloc[0]
    assert bcc_pfs["ci_low_status"] == "numeric"
    assert bcc_pfs["ci_basis"] == "reported"

    chordoma_os = df[
        (df["cancer_code"] == "SARC_CHOR")
        & (df["regimen"] == "PD-1")
        & (df["metric"] == "OS")
        & (df["value_status"] == "not_reached")
    ].iloc[0]
    assert chordoma_os["ci_low_status"] == "numeric"
    assert chordoma_os["ci_high_status"] == "NR"

    coad_blend = df[
        (df["cancer_code"] == "COAD")
        & (df["metric"] == "ORR")
        & (df["value_basis"] == "derived_blend")
    ].iloc[0]
    assert coad_blend["ci_basis"] == "not_applicable"


def test_ici_source_locator_audit_is_complete_and_matches_estimates():
    estimates = ici.cancer_ici_response_estimates_df()
    audit = ici.cancer_ici_source_locator_audit_df()

    assert audit["estimate_id"].is_unique
    assert set(audit["estimate_id"]) == set(estimates["estimate_id"])
    assert audit["source_document_kind"].notna().all()
    assert audit["audited_on"].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    cited = audit["ref"].fillna("").astype(str).str.strip().ne("")
    assert audit.loc[cited, "source_document_url"].fillna("").str.strip().ne("").all()
    assert audit.loc[cited, "source_document_kind"].ne("missing_citation").all()

    joined = estimates[["estimate_id", "ref", "source_locator", "source_locator_status"]].merge(
        audit[["estimate_id", "ref", "source_locator", "source_locator_status"]],
        on="estimate_id",
        suffixes=("_estimate", "_audit"),
        validate="one_to_one",
    )
    for column in ("ref", "source_locator", "source_locator_status"):
        assert joined[f"{column}_estimate"].equals(joined[f"{column}_audit"])

    located = audit["source_locator_status"].isin(
        {"located_unverified", "source_section", "verified"}
    )
    assert audit.loc[located, "source_locator"].astype(str).str.strip().ne("").all()
    citation_only = audit["source_locator_status"].eq("citation_only")
    assert audit.loc[citation_only, "source_locator"].fillna("").str.strip().eq("").all()


def test_ici_source_audit_corrections_and_computed_ci():
    df = ici.cancer_ici_response_estimates_df().set_index("estimate_id")

    meningioma_os = df.loc["ICI-784f9b8f57-01"]
    assert meningioma_os["value"] == 20.2
    assert meningioma_os["ci_low"] == 14.8
    assert meningioma_os["ci_high"] == 25.8
    assert meningioma_os["source_locator"] == "Supplementary Figure 2"

    kirc_hr = df.loc["ICI-7be940d63d-01"]
    assert kirc_hr["metric"] == "PFS_HR"
    assert kirc_hr["unit"] == "hazard_ratio"
    assert kirc_hr["value"] == 1.03

    assert df.loc["ICI-cb72c7e2fd-01", "metric"] == "CBR"
    assert df.loc["ICI-c8a3a23b71-01", "metric"] == "CBR"
    assert df.loc["ICI-727d3fda5e-01", "metric"] == "UNCONFIRMED_ORR"

    bcc_crr = df.loc["ICI-4398a82d1e-01"]
    assert bcc_crr["value"] == 3.7
    assert bcc_crr["ci_basis"] == "computed_wilson"
    assert bcc_crr["ci_low"] == 1.0
    assert bcc_crr["ci_high"] == 12.5

    unavailable_dor = df.loc["ICI-de9e5a30cc-01"]
    assert unavailable_dor["value_status"] == "not_reported"
    assert unavailable_dor["ci_basis"] == "not_applicable"


def test_keynote051_nbl_anchor_uses_treated_response_denominator():
    estimate_id = "ICI-e9a3cf4215-01"
    estimates = ici.cancer_ici_response_estimates_df().set_index("estimate_id")
    row = estimates.loc[estimate_id]

    assert row["cancer_code"] == "NBL"
    assert row["trial_name"] == "KEYNOTE-051"
    assert row["source_n"] == 11
    assert row["metric_n"] == 11
    assert row["responders"] == 0
    assert row["value"] == 0.0
    assert row["value_basis"] == "computed_from_counts"
    assert row["ci_basis"] == "computed_clopper_pearson"
    assert row["ci_low"] == 0.0
    expected_high = 100 * (1 - math.pow(0.025, 1 / 11))
    assert math.isclose(row["ci_high"], expected_high, abs_tol=5e-5)
    assert "n=80 is PD-L1 screening only" in row["note"]

    anchor = ici.cancer_ici_response_record("NBL")
    assert anchor["source_estimate_id"] == estimate_id
    assert anchor["response_denominator"] == 11
    assert anchor["response_numerator"] == 0
    assert anchor["response_ci_basis"] == "computed_clopper_pearson"
    assert anchor["evidence_type"] == "computed_from_counts"
    assert ici.cancer_ici_response("NBL") == 0.0

    apd1_anchor = apd1.cancer_apd1_response_record("NBL")
    assert apd1_anchor["source_estimate_id"] == estimate_id
    assert apd1_anchor["response_denominator"] == 11
    assert apd1.cancer_apd1_response("NBL") == 0.0

    pooled = ici.pooled_ici_response("NBL", regimen="PD-1", metric="ORR")
    assert pooled["pooled_pct"] == 0.0
    assert pooled["n_total"] == 11
    assert pooled["responders_total"] == 0

    audit = ici.cancer_ici_source_locator_audit_df().set_index("estimate_id").loc[estimate_id]
    assert audit["source_locator"] == "Table 1; Table 4 footnote; Results"
    assert audit["source_locator_status"] == "verified"


def test_aml_combination_stays_out_of_monotherapy_anchor_and_mpn_scope_is_explicit():
    estimates = ici.cancer_ici_response_estimates_df().set_index("estimate_id")

    aml = estimates.loc["ICI-8b126eef2c-01"]
    assert aml["cancer_code"] == "LAML"
    assert aml["regimen"] == "PD-1+HMA"
    assert aml["role"] == "alternate"
    assert aml["trial_nct"] == "NCT02397720"
    assert aml["source_n"] == 70
    assert aml["metric_n"] == 70
    assert aml["responders"] == 23
    assert aml["value"] == 33.0
    assert aml["ci_basis"] == "computed_wilson"
    assert "15 CR/CRi + 1 PR + 7 hematologic improvements" in aml["note"]

    anchors = ici.cancer_ici_response_df()
    assert "LAML" not in set(anchors["cancer_code"])
    assert ici.cancer_ici_response("LAML") is None
    assert ici.cancer_ici_response_record("LAML") is None
    assert ici.cancer_ici_response("LAML", regimen="PD-1", inherit=False) is None
    assert ici.cancer_ici_response_record("LAML", regimen="PD-1", inherit=False) is None
    assert apd1.cancer_apd1_response("LAML") is None

    monotherapy = ici.pooled_ici_response("LAML", regimen="PD-1", metric="ORR")
    assert monotherapy["n_studies"] == 0
    assert monotherapy["pooled_pct"] is None
    combination = ici.pooled_ici_response("LAML", regimen="PD-1+HMA", metric="ORR")
    assert combination["n_studies"] == 1
    assert combination["n_total"] == 70
    assert combination["responders_total"] == 23
    assert combination["pooled_pct"] == 32.9
    primary_only = ici.pooled_ici_response(
        "LAML", regimen="PD-1+HMA", metric="ORR", include_alternates=False
    )
    assert primary_only["n_studies"] == 0
    assert ici.REGIMEN_CLASSES["PD-1+HMA"] == "anti_pd1_hma_combination"

    mpn = estimates.loc["ICI-4b4990885c-01"]
    assert mpn["role"] == "alternate"
    assert mpn["value"] == 0.0
    assert mpn["responders"] == 0
    assert mpn["metric_n"] == 10
    assert "myelofibrosis" in mpn["source_population_label"].lower()
    assert "must not become a representative anchor" in mpn["note"]
    assert ici.cancer_ici_response("MPN", inherit=False) is None

    audit = ici.cancer_ici_source_locator_audit_df().set_index("estimate_id")
    assert audit.loc["ICI-8b126eef2c-01", "source_locator_status"] == "verified"
    assert audit.loc["ICI-4b4990885c-01", "source_locator_status"] == "verified"


def test_apd1_anchor_table_uses_same_evidence_schema_for_fallback_targets():
    df = apd1.cancer_apd1_response_df()
    assert {
        "response_metric",
        "response_numerator",
        "response_denominator",
        "therapy_regimen_class",
        "evidence_type",
        "source_scope",
    } <= set(df.columns)

    acc = df[df["cancer_code"] == "ACC"].iloc[0]
    assert acc["drug_target"] == "PD-1+CTLA-4"
    assert acc["therapy_regimen_class"] == "anti_pd1_ctla4_combination"
    assert acc["response_numerator"] == 3
    assert acc["response_denominator"] == 21
    assert acc["evidence_type"] == "direct_reported"


def test_per_regimen_and_pin():
    # melanoma carries both anti-PD-1 mono and the ipi+nivo doublet, as distinct sources
    per = ici.cancer_ici_response("SKCM", fallback=False)
    assert per["PD-1"] > 0 and per["PD-1+CTLA-4"] > per["PD-1"]
    assert ici.cancer_ici_response("SKCM", regimen="PD-1+CTLA-4") == per["PD-1+CTLA-4"]


def test_fallback_prefers_pd1_then_pdl1():
    # SKCM has anti-PD-1 -> fallback picks PD-1, not the higher combo value.
    assert ici.cancer_ici_regimen("SKCM") == "PD-1"
    assert ici.cancer_ici_response("SKCM") == ici.cancer_ici_response("SKCM", regimen="PD-1")
    # SARC_ASPS has only anti-PD-L1 -> fallback resolves to PD-L1.
    assert ici.cancer_ici_regimen("SARC_ASPS") == "PD-L1"
    assert ici.cancer_ici_response("SARC_ASPS") == ici.cancer_ici_response(
        "SARC_ASPS", regimen="PD-L1"
    )


def test_mpnst_pd1_response_row():
    assert ici.cancer_ici_regimen("SARC_MPNST") == "PD-1"
    assert ici.cancer_ici_response("SARC_MPNST") == 12.5

    record = ici.cancer_ici_response_record("SARC_MPNST")
    assert record["resolved_cancer_code"] == "SARC_MPNST"
    assert record["selected_regimen"] == "PD-1"
    assert record["source_anchor"] == "PMID:41760889"
    assert record["response_numerator"] == 1
    assert record["response_denominator"] == 8

    estimates = ici.cancer_ici_response_estimates_df()
    rows = estimates[
        (estimates["cancer_code"] == "SARC_MPNST") & (estimates["regimen"] == "PD-1")
    ].set_index("metric")
    assert rows.loc["ORR", "responders"] == 1
    assert rows.loc["ORR", "metric_n"] == 8
    assert rows.loc["CBR", "value"] == 12.5
    assert rows.loc["DCR", "responders"] == 5
    assert rows.loc["PFS", "value"] == 3.9
    assert rows.loc["PFS", "ci_high"] == 8.1
    assert rows.loc["OS", "value"] == 7.3
    assert rows.loc["OS", "ci_high"] == 26.3


def test_ucec_pole_pd1_anchor_is_direct_not_bulk_parent_blend():
    assert apd1.cancer_apd1_response("UCEC_POLE") == 100.0
    assert ici.cancer_ici_response("UCEC_POLE") == 100.0
    assert ici.cancer_ici_response("UCEC_POLE", inherit=False) == 100.0

    record = ici.cancer_ici_response_record("UCEC_POLE")
    assert record["requested_cancer_code"] == "UCEC_POLE"
    assert record["resolved_cancer_code"] == "UCEC_POLE"
    assert record["inheritance_kind"] == "direct"
    assert record["is_inherited_evidence"] is False
    assert record["confidence"] == "low"
    assert record["source_anchor"] == "PMID:27159395"
    assert record["response_numerator"] == 1
    assert record["response_denominator"] == 1
    assert record["orr_pct"] > ici.cancer_ici_response("UCEC")


def test_maps_and_alias():
    assert ici.cancer_ici_response("melanoma") == ici.cancer_ici_response("SKCM")
    full = ici.cancer_ici_response()
    pdl1 = ici.cancer_ici_response(regimen="PD-L1")
    assert len(full) > len(pdl1) >= 10
    assert all(isinstance(v, float) for v in full.values())


def test_whole_table_per_regimen_mapping():
    # cancer_type=None with fallback=False -> {code: {regimen: orr}} for every cancer.
    per = ici.cancer_ici_response(fallback=False)
    assert isinstance(per["SKCM"], dict)
    assert per["SKCM"] == {"PD-1": 42.0, "PD-1+CTLA-4": 57.6}
    # single-regimen cancers carry a one-entry mapping
    assert set(per["SARC_ASPS"]) == {"PD-L1"}
    # the PD-L1 members match the pinned PD-L1 map
    assert {c for c, m in per.items() if "PD-L1" in m} == set(
        ici.cancer_ici_response(regimen="PD-L1")
    )


def test_crc_msi_ici_is_single_source_scope_row():
    full = ici.cancer_ici_response()
    per = ici.cancer_ici_response(fallback=False)
    assert full["CRC_MSI"] == 43.8
    assert per["CRC_MSI"] == {"PD-1": 43.8, "PD-1+CTLA-4": 55.0}
    assert "COAD_MSI" not in full
    assert "READ_MSI" not in full
    assert ici.cancer_ici_response("COAD_MSI") == full["CRC_MSI"]
    assert ici.cancer_ici_response("READ_MSI", regimen="PD-1") == full["CRC_MSI"]
    assert ici.cancer_ici_response("COAD_MSI", fallback=False) == per["CRC_MSI"]
    assert ici.cancer_ici_response("READ_MSI", inherit=False) is None
    assert ici.cancer_ici_regimen("READ_MSI") == "PD-1"

    inherited = ici.cancer_ici_response(include_inherited=True)
    inherited_per = ici.cancer_ici_response(fallback=False, include_inherited=True)
    assert inherited["COAD_MSI"] == full["CRC_MSI"]
    assert inherited["READ_MSI"] == full["CRC_MSI"]
    assert inherited_per["COAD_MSI"] == per["CRC_MSI"]
    assert inherited_per["READ_MSI"] == per["CRC_MSI"]
    assert "READ_MSI" not in ici.cancer_ici_response(include_inherited=True, inherit=False)


def test_impower110_nsclc_pdl1_is_not_direct_luad_or_lusc():
    pdl1 = ici.cancer_ici_response(regimen="PD-L1")
    assert pdl1["NSCLC"] == 38.3
    assert "LUAD" not in pdl1
    assert "LUSC" not in pdl1

    assert ici.cancer_ici_response("LUAD", regimen="PD-L1", inherit=False) is None
    assert ici.cancer_ici_response("LUSC", regimen="PD-L1", inherit=False) is None
    assert ici.cancer_ici_response("LUAD", regimen="PD-L1") == 38.3
    assert ici.cancer_ici_response("LUSC", regimen="PD-L1") == 38.3
    assert ici.cancer_ici_response("LUAD") == 19.0
    assert ici.cancer_ici_response("LUSC") == 20.0

    record = ici.cancer_ici_response_record("LUSC", regimen="PD-L1")
    assert record["requested_cancer_code"] == "LUSC"
    assert record["resolved_cancer_code"] == "NSCLC"
    assert record["inheritance_kind"] == "ancestor"
    assert record["is_inherited_evidence"] is True
    assert record["response_denominator"] == 107
    assert record["response_numerator"] == 41
    assert record["source_scope"] == "aggregate_source"
    assert "all histologies" in record["endpoint_population"]


def test_btc_ici_is_single_pan_biliary_source_scope_row():
    full = ici.cancer_ici_response()
    per = ici.cancer_ici_response(fallback=False)
    assert full["BTC"] == 5.8
    assert per["BTC"] == {"PD-1": 5.8, "PD-L1": 4.8}
    assert "CHOL" not in full
    assert "GBC" not in full
    assert ici.cancer_ici_response("CHOL") == full["BTC"]
    assert ici.cancer_ici_response("GBC") == full["BTC"]
    assert ici.cancer_ici_response("GBC", regimen="PD-L1") == 4.8
    assert ici.cancer_ici_response("GBC", inherit=False) is None

    inherited_pdl1 = ici.cancer_ici_response(regimen="PD-L1", include_inherited=True)
    assert inherited_pdl1["GBC"] == 4.8
    assert inherited_pdl1["CHOL"] == 4.8


def test_sgc_ici_is_single_pan_salivary_source_scope_row():
    full = ici.cancer_ici_response()
    per = ici.cancer_ici_response(fallback=False)
    assert full["SGC"] == 4.6
    assert per["SGC"] == {"PD-1": 4.6}
    assert "ACINIC" not in full
    assert ici.cancer_ici_response("ACINIC") == full["SGC"]
    assert ici.cancer_ici_response("ACINIC", inherit=False) is None
    # ADCC has direct combination data, but pinned anti-PD-1 uses the pan-salivary row.
    assert full["ADCC"] == 6.0
    assert ici.cancer_ici_response("ADCC", regimen="PD-1") == full["SGC"]
    assert ici.cancer_ici_response("ADCC", regimen="PD-1", inherit=False) is None


def test_net_nonpancreatic_ici_is_single_source_scope_row():
    full = ici.cancer_ici_response()
    per = ici.cancer_ici_response(fallback=False)
    assert full["NET_NONPANCREATIC"] == 0.0
    assert per["NET_NONPANCREATIC"] == {"PD-1+CTLA-4": 0.0}
    assert "NET_LUNG" not in full
    assert "NET_MIDGUT" not in full
    assert "NET_RECTAL" not in full
    assert ici.cancer_ici_response("NET_LUNG") == full["NET_NONPANCREATIC"]
    assert ici.cancer_ici_response("NET_MIDGUT") == full["NET_NONPANCREATIC"]
    assert ici.cancer_ici_response("NET_RECTAL") == full["NET_NONPANCREATIC"]
    assert ici.cancer_ici_response("NET_LUNG", inherit=False) is None
    assert ici.cancer_ici_response("NET_PANCREAS") == 11.0


def test_extrapulmonary_g3_nen_ici_is_not_lung_lcnec():
    full = ici.cancer_ici_response()
    per = ici.cancer_ici_response(fallback=False)
    assert full["NEN_EXTRAPULMONARY_HG"] == 3.4
    assert per["NEN_EXTRAPULMONARY_HG"] == {"PD-1": 3.4}
    assert full["NEC_LUNG_LARGECELL"] == 29.4
    assert ici.cancer_ici_response("extrapulmonary G3 NEN") == 3.4


def test_crc_msi_ici_record_preserves_inheritance_metadata():
    record = ici.cancer_ici_response_record("COAD_MSI")
    assert record["requested_cancer_code"] == "COAD_MSI"
    assert record["resolved_cancer_code"] == "CRC_MSI"
    assert record["inheritance_kind"] == "source_scope"
    assert record["is_inherited_evidence"] is True
    assert record["regimen"] == "PD-1"
    assert record["selected_regimen"] == "PD-1"
    assert record["orr_pct"] == 43.8
    assert record["response_numerator"] == 67
    assert record["response_denominator"] == 153
    assert record["response_ci_low"] == 35.8
    assert record["response_ci_high"] == 52.0
    assert record["source_anchor"] == "PMID:33264544"
    assert record["source_scope"] == "aggregate_source"
    assert record["endpoint_population"] == (
        "first-line metastatic MSI-H/dMMR colorectal (pembrolizumab arm)"
    )

    per_regimen = ici.cancer_ici_response_record("READ_MSI", fallback=False)
    assert set(per_regimen) == {"PD-1", "PD-1+CTLA-4"}
    assert per_regimen["PD-1"]["requested_cancer_code"] == "READ_MSI"
    assert per_regimen["PD-1"]["resolved_cancer_code"] == "CRC_MSI"
    assert per_regimen["PD-1+CTLA-4"]["response_denominator"] == 119
    assert per_regimen["PD-1+CTLA-4"]["source_anchor"] == "PMID:29355075"

    assert ici.cancer_ici_response_record("READ_MSI", inherit=False) is None
    assert ici.cancer_ici_response_record("READ_MSI", fallback=False, inherit=False) == {}

    bulk = ici.cancer_ici_response_record(include_inherited=True)
    assert bulk["COAD_MSI"]["requested_cancer_code"] == "COAD_MSI"
    assert bulk["COAD_MSI"]["resolved_cancer_code"] == "CRC_MSI"
    assert bulk["COAD_MSI"]["inheritance_kind"] == "source_scope"
    assert bulk["COAD_MSI"]["is_inherited_evidence"] is True

    bulk_per_regimen = ici.cancer_ici_response_record(fallback=False, include_inherited=True)
    assert set(bulk_per_regimen["READ_MSI"]) == {"PD-1", "PD-1+CTLA-4"}
    assert bulk_per_regimen["READ_MSI"]["PD-1"]["resolved_cancer_code"] == "CRC_MSI"


def test_btc_ici_record_preserves_inheritance_metadata():
    record = ici.cancer_ici_response_record("GBC")
    assert record["requested_cancer_code"] == "GBC"
    assert record["resolved_cancer_code"] == "BTC"
    assert record["inheritance_kind"] == "source_scope"
    assert record["is_inherited_evidence"] is True
    assert record["regimen"] == "PD-1"
    assert record["orr_pct"] == 5.8
    assert record["response_numerator"] == 6
    assert record["response_denominator"] == 104
    assert record["source_scope"] == "aggregate_source"
    assert (
        record["endpoint_population"]
        == "advanced biliary tract cancer, prior-treated pan-biliary cohort"
    )


def test_sgc_ici_record_preserves_inheritance_metadata():
    record = ici.cancer_ici_response_record("ACINIC")
    assert record["requested_cancer_code"] == "ACINIC"
    assert record["resolved_cancer_code"] == "SGC"
    assert record["inheritance_kind"] == "source_scope"
    assert record["is_inherited_evidence"] is True
    assert record["regimen"] == "PD-1"
    assert record["orr_pct"] == 4.6
    assert record["response_numerator"] == 5
    assert record["response_denominator"] == 109
    assert record["response_ci_low"] == 1.5
    assert record["response_ci_high"] == 10.4
    assert record["source_scope"] == "aggregate_source"
    assert (
        record["endpoint_population"] == "previously treated advanced salivary gland carcinoma "
        "(pan-salivary source-scope estimate)"
    )

    adcc_pd1 = ici.cancer_ici_response_record("ADCC", regimen="PD-1")
    assert adcc_pd1["resolved_cancer_code"] == "SGC"
    assert adcc_pd1["inheritance_kind"] == "source_scope"


def test_net_nonpancreatic_ici_record_preserves_inheritance_metadata():
    record = ici.cancer_ici_response_record("NET_LUNG")
    assert record["requested_cancer_code"] == "NET_LUNG"
    assert record["resolved_cancer_code"] == "NET_NONPANCREATIC"
    assert record["inheritance_kind"] == "source_scope"
    assert record["is_inherited_evidence"] is True
    assert record["regimen"] == "PD-1+CTLA-4"
    assert record["orr_pct"] == 0.0
    assert record["response_numerator"] == 0
    assert record["response_denominator"] == 14
    assert record["response_ci_low"] == 0
    assert record["response_ci_high"] == 23
    assert record["source_scope"] == "aggregate_source"
    assert (
        record["endpoint_population"]
        == "low/intermediate-grade nonpancreatic NET (pooled; not site-isolated)"
    )

    midgut = ici.cancer_ici_response_record("NET_MIDGUT")
    assert midgut["resolved_cancer_code"] == "NET_NONPANCREATIC"
    assert midgut["inheritance_kind"] == "source_scope"


def test_extrapulmonary_g3_nen_record_is_direct_context_aggregate():
    record = ici.cancer_ici_response_record("NEN_G3_EXTRAPULMONARY")
    assert record["requested_cancer_code"] == "NEN_EXTRAPULMONARY_HG"
    assert record["resolved_cancer_code"] == "NEN_EXTRAPULMONARY_HG"
    assert record["inheritance_kind"] == "direct"
    assert record["is_inherited_evidence"] is False
    assert record["orr_pct"] == 3.4
    assert record["response_numerator"] == 1
    assert record["response_denominator"] == 29
    assert record["response_ci_low"] == 0.1
    assert record["response_ci_high"] == 17.8
    assert "extrapulmonary" in record["endpoint_population"]


def test_resolve_ici_response_source_reports_direct_proxy_and_missing():
    direct = ici.resolve_ici_response_source("SKCM")
    assert direct["requested_cancer_code"] == "SKCM"
    assert direct["resolved_cancer_code"] == "SKCM"
    assert direct["inheritance_kind"] == "direct"
    assert direct["is_inherited_evidence"] is False
    assert direct["selected_regimen"] == "PD-1"
    assert direct["available_regimens"] == ("PD-1", "PD-1+CTLA-4")
    assert direct["has_ici_response_source"] is True
    assert direct["source_anchor"] == "PMID:28889792"

    proxy = ici.resolve_ici_response_source("COAD_MSI")
    assert proxy["requested_cancer_code"] == "COAD_MSI"
    assert proxy["resolved_cancer_code"] == "CRC_MSI"
    assert proxy["inheritance_kind"] == "source_scope"
    assert proxy["is_inherited_evidence"] is True
    assert proxy["selected_regimen"] == "PD-1"
    assert proxy["available_regimens"] == ("PD-1", "PD-1+CTLA-4")
    assert proxy["source_anchor"] == "PMID:33264544"
    assert proxy["source_scope"] == "aggregate_source"

    per_regimen = ici.resolve_ici_response_source("READ_MSI", fallback=False)
    assert per_regimen["resolved_cancer_code"] == "CRC_MSI"
    assert per_regimen["selected_regimen"] is None
    assert per_regimen["available_regimens"] == ("PD-1", "PD-1+CTLA-4")

    missing = ici.resolve_ici_response_source("HCL")
    assert missing == {
        "requested_cancer_code": "HCL",
        "resolved_cancer_code": None,
        "inheritance_kind": "missing",
        "is_inherited_evidence": False,
        "selected_regimen": None,
        "available_regimens": (),
        "has_ici_response_source": False,
    }


def test_ici_response_record_whole_table_matches_value_maps():
    records = ici.cancer_ici_response_record()
    values = ici.cancer_ici_response()
    assert set(records) == set(values)
    assert {code: record["orr_pct"] for code, record in records.items()} == values
    assert "COAD_MSI" not in records
    assert records["CRC_MSI"]["inheritance_kind"] == "direct"

    pdl1_records = ici.cancer_ici_response_record(regimen="PD-L1")
    pdl1_values = ici.cancer_ici_response(regimen="PD-L1")
    assert {code: record["orr_pct"] for code, record in pdl1_records.items()} == pdl1_values


def test_parent_code_helper_treats_nan_parent_as_missing():
    registry = ici.cancer_type_registry().set_index("code")
    assert ici._parent_code("CRC", registry) is None


def test_regimen_maps_cached():
    # _regimen_maps is memoized (same object back from the cache).
    assert ici._regimen_maps() is ici._regimen_maps()
