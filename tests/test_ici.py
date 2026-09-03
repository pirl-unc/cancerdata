# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math

import pandas as pd
import pytest

from oncoref import apd1, ici, tmb


def test_regimens_and_table():
    assert ici.ici_regimens() == ("PD-1", "PD-L1", "PD-1+CTLA-4")
    df = ici.cancer_ici_response_df()
    assert {"cancer_code", "regimen", "orr_pct"} <= set(df.columns)
    # Curated gap rows carry no ORR and no regimen — the regimen vocabulary applies to
    # rows that actually anchor a value. Exempt only the declared gaps, so a stray
    # blank cannot slip past this check.
    gap_codes = set(ici._ICI_EVIDENCE_OVERRIDES)
    valued = df[~df["cancer_code"].isin(gap_codes)]
    assert valued["orr_pct"].notna().all()
    # all three regimens are actually present (not just PD-1)
    assert set(valued["regimen"]) == {"PD-1", "PD-L1", "PD-1+CTLA-4"}
    assert (valued["regimen"] == "PD-L1").sum() >= 10  # anti-PD-L1 is well-represented
    assert df.loc[df["cancer_code"].isin(gap_codes), "regimen"].isna().all()


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


def test_audited_ici_gap_is_distinguishable_from_an_unreviewed_code():
    """A curated gap row must resolve, not vanish.

    Before this contract existed the response table could only express "has a value";
    a blank-ORR row was dropped by the value maps, so a reviewed "no defensible
    aggregate exists" was indistinguishable from a code nobody had curated.
    """
    reviewed = ici.resolve_ici_response_source("CRC")
    assert reviewed["inheritance_kind"] == "direct_missing"
    assert reviewed["has_ici_response_source"] is True
    assert reviewed["resolved_cancer_code"] == "CRC"
    assert reviewed["is_inherited_evidence"] is False
    assert reviewed["available_regimens"] == ()
    assert reviewed["selected_regimen"] is None
    assert reviewed["evidence_type"] == "unknown"
    assert reviewed["source_scope"] == "subtype_sources_not_aggregated"
    assert reviewed["missing_reason"] == "response_is_mmr_stratified_not_aggregate"

    # An unreviewed code keeps the original bare-miss contract.
    unreviewed = ici.resolve_ici_response_source("HCL")
    assert unreviewed["inheritance_kind"] == "missing"
    assert unreviewed["has_ici_response_source"] is False
    assert "missing_reason" not in unreviewed


def test_audited_ici_gap_is_reported_on_every_surface():
    """The gap must survive inherit=False and reach the record accessor.

    tmb.resolve_tmb_source / tmb.cancer_tmb_record both report an audited gap
    regardless of ``inherit``; ICI must match or the "same contract" claim in
    docs/api.md is false.
    """
    for kwargs in ({}, {"inherit": False}):
        record = ici.resolve_ici_response_source("CRC", **kwargs)
        assert record["inheritance_kind"] == "direct_missing", kwargs
        assert record["has_ici_response_source"] is True, kwargs
        assert record["missing_reason"] == "response_is_mmr_stratified_not_aggregate"

    # The record surface distinguishes a reviewed gap from an uncurated code.
    gap_record = ici.cancer_ici_response_record("CRC")
    assert gap_record is not None
    assert gap_record["inheritance_kind"] == "direct_missing"
    assert gap_record["missing_reason"] == "response_is_mmr_stratified_not_aggregate"
    assert ici.cancer_ici_response_record("HCL") is None

    # A gap names no regimen, so the per-regimen view is empty rather than inherited.
    assert ici.cancer_ici_response("CRC", fallback=False) == {}
    assert ici.cancer_ici_response_record("CRC", fallback=False) == {}


def test_subtype_stratified_aggregates_do_not_report_a_pooled_orr():
    """CRC/RCC/BRCA/SARC response is subtype-determined, so the umbrella has no ORR."""
    expected = {
        "CRC": "response_is_mmr_stratified_not_aggregate",
        "RCC": "no_supported_aggregate_orr",
        "BRCA": "response_is_receptor_subtype_stratified",
        "SARC": "response_is_histology_stratified",
    }
    for code, reason in expected.items():
        assert ici.cancer_ici_response(code) is None
        record = ici.resolve_ici_response_source(code)
        assert record["inheritance_kind"] == "direct_missing"
        assert record["missing_reason"] == reason
        assert record["source_scope"] == "subtype_sources_not_aggregated"

    # The stratified children each keep their own curated anchor.
    assert ici.cancer_ici_response("CRC_MSI") == 43.8
    assert ici.cancer_ici_response("KIRC") == 25.0
    assert ici.cancer_ici_response("BRCA_Basal") is not None
    assert ici.cancer_ici_response("SARC_UPS") == 23.0

    # Gap rows stay out of the bulk value map and out of the regimen maps.
    bulk = ici.cancer_ici_response()
    assert not ({"CRC", "RCC", "BRCA", "SARC"} & set(bulk))


def test_ici_gap_row_blocks_inheritance_instead_of_borrowing_an_ancestor():
    """A reviewed gap outranks ancestor evidence, mirroring blank-value TMB rows.

    STAD_MSI is the case that actually exercises this: its parent STAD carries an
    all-comer gastric anchor, so without the guard the resolver would hand MSI-H
    disease the all-comer ORR.
    """
    registry = ici.cancer_type_registry().set_index("code")
    assert registry.loc["STAD_MSI", "parent_code"] == "STAD"
    assert ici.cancer_ici_response("STAD") == 12.0

    # Every lookup path refuses to inherit through the gap.
    assert ici.cancer_ici_response("STAD_MSI") is None
    assert ici.cancer_ici_response("STAD_MSI", fallback=False) == {}
    resolved = ici.resolve_ici_response_source("STAD_MSI")
    assert resolved["inheritance_kind"] == "direct_missing"
    assert resolved["resolved_cancer_code"] == "STAD_MSI"
    assert resolved["missing_reason"] == "no_supported_subtype_orr"

    # TMB and ICI now agree that STAD_MSI is audited-unknown rather than half-audited.
    assert tmb.cancer_tmb("STAD_MSI") is None
    assert tmb.resolve_tmb_source("STAD_MSI")["inheritance_kind"] == "direct_missing"

    # A sibling without a gap row still inherits normally.
    assert ici.cancer_ici_response("STAD_CIN") == 12.0


def test_only_declared_codes_may_carry_a_blank_orr():
    """An accidental blank must raise, not become an 'audited' gap.

    The declaration is checked against the blank value itself, not against the
    estimates join: a row that keeps its estimate while losing its value still joins
    cleanly, so a join-keyed check would wave it through.
    """
    declared = ici._ICI_EVIDENCE_OVERRIDES
    anchors = ici.get_data("cancer-ici-response").copy()

    # The shipped table is valid only because its blank rows are declared gaps.
    ici.response_anchor_evidence_df(anchors, value_col="orr_pct", gap_overrides=declared)

    # The realistic slip: clearing the value on a row that already has an estimate.
    # The join still succeeds, so only the blank-value check can catch this.
    cleared = anchors.copy()
    cleared.loc[cleared["cancer_code"] == "LUAD", "orr_pct"] = float("nan")
    with pytest.raises(ValueError, match="only allowed for declared audited gaps"):
        ici.response_anchor_evidence_df(cleared, value_col="orr_pct", gap_overrides=declared)

    # And the slip with no estimate to join against.
    slip = anchors[anchors["cancer_code"] == "LUAD"].iloc[[0]].copy()
    slip["orr_pct"] = float("nan")
    slip["cancer_code"] = "HCL"
    perturbed = pd.concat([anchors, slip], ignore_index=True)
    with pytest.raises(ValueError, match="only allowed for declared audited gaps"):
        ici.response_anchor_evidence_df(perturbed, value_col="orr_pct", gap_overrides=declared)

    # Declaring the code is not enough either: a gap row must not name a regimen.
    with pytest.raises(ValueError, match="must leave regimen blank"):
        ici.response_anchor_evidence_df(
            perturbed,
            value_col="orr_pct",
            gap_overrides={**declared, "HCL": {"source_scope": "s", "missing_reason": "r"}},
        )

    # A code cannot be both valued and an audited gap. COAD keeps its valued anchor
    # (which has a backing estimate), so only the contradiction itself can raise.
    blank_coad = anchors[anchors["cancer_code"] == "COAD"].iloc[[0]].copy()
    blank_coad["orr_pct"] = float("nan")
    blank_coad["regimen"] = None
    with pytest.raises(ValueError, match="both valued and an audited gap"):
        ici.response_anchor_evidence_df(
            pd.concat([anchors, blank_coad], ignore_index=True),
            value_col="orr_pct",
            gap_overrides={**declared, "COAD": {"source_scope": "s", "missing_reason": "r"}},
        )

    # Two gap rows for one code would collapse the code-keyed lookup to whichever came
    # last; the merge's one_to_one validation already rejects it (duplicate NaN
    # regimen keys count as duplicates in the left frame).
    twice = pd.concat(
        [anchors, anchors[anchors["cancer_code"] == "CRC"].iloc[[0]].copy()], ignore_index=True
    )
    with pytest.raises(pd.errors.MergeError, match="not a one-to-one merge"):
        ici.response_anchor_evidence_df(twice, value_col="orr_pct", gap_overrides=declared)

    # The shared helper defaults to allowing no gaps at all, so the aPD-1 table keeps
    # the strict contract.
    with pytest.raises(ValueError, match="only allowed for declared audited gaps"):
        ici.response_anchor_evidence_df(anchors, value_col="orr_pct")


def test_audited_gap_agrees_across_every_lookup_surface():
    """The gap guard lives in four resolution paths; they must not drift apart."""
    for code in ici._ICI_EVIDENCE_OVERRIDES:
        assert ici.cancer_ici_response(code) is None, code
        assert ici.cancer_ici_response(code, inherit=False) is None, code
        assert ici.cancer_ici_response(code, fallback=False) == {}, code
        assert ici.cancer_ici_response(code, regimen="PD-1") is None, code

        record = ici.cancer_ici_response_record(code)
        assert record is not None and record["inheritance_kind"] == "direct_missing", code
        assert ici.cancer_ici_response_record(code, fallback=False) == {}, code

        for kwargs in ({}, {"inherit": False}, {"regimen": "PD-1"}, {"fallback": False}):
            resolved = ici.resolve_ici_response_source(code, **kwargs)
            assert resolved["inheritance_kind"] == "direct_missing", (code, kwargs)
            assert resolved["has_ici_response_source"] is True, (code, kwargs)
            assert resolved["missing_reason"], (code, kwargs)


def test_gap_rows_do_not_publish_fabricated_evidence_fields():
    """A gap row names no regimen and cites no evidence, so derived fields stay empty."""
    df = ici.cancer_ici_response_df()
    gaps = df[df["orr_pct"].isna()]
    assert set(gaps["cancer_code"]) == {"CRC", "RCC", "BRCA", "SARC", "STAD_MSI"}
    assert gaps["regimen"].isna().all()
    assert gaps["therapy_regimen_class"].isna().all()
    assert gaps["evidence_source_code"].isna().all()
    assert gaps["response_metric"].isna().all()
    assert gaps["source_estimate_id"].isna().all()
    assert (gaps["evidence_type"] == "unknown").all()
    assert not gaps["is_direct_cancer_code_evidence"].any()
    assert gaps["missing_reason"].notna().all()


def test_gap_note_citations_resolve_to_real_anchor_rows():
    """The notes column is the audit trail, so its citations must be real.

    A gap note that names a trial or PMID the table does not actually anchor is a
    curation error, not just prose drift — and prose is exactly what no other test
    checks.
    """
    import re

    df = ici.cancer_ici_response_df()
    valued = df[df["orr_pct"].notna()]
    known_refs = set(valued["pmid_doi"].dropna().astype(str))
    gaps = df[df["cancer_code"].isin(ici._ICI_EVIDENCE_OVERRIDES)]
    assert not gaps.empty

    for _, row in gaps.iterrows():
        code = row["cancer_code"]
        cited = set(re.findall(r"PMID:\d+|DOI:[^\s,;)]+", str(row["notes"])))
        assert cited, f"{code} gap note cites no source"
        unknown = cited - known_refs
        assert not unknown, f"{code} note cites refs absent from the anchor table: {unknown}"


def test_gap_note_quoted_values_match_the_curated_anchors():
    """Percentages quoted in a gap note must equal the rows they describe."""

    def orr(code, regimen="PD-1"):
        df = ici.cancer_ici_response_df()
        hit = df[(df["cancer_code"] == code) & (df["regimen"] == regimen)]
        return float(hit["orr_pct"].iloc[0])

    # CRC note: "CRC_MSI 43.8%"
    assert orr("CRC_MSI") == 43.8
    # RCC note: "KIRC 25.0% ... 42.0% ... KIRP 28.8% and KICH 9.5% ... spanning 9.5% to 42%"
    assert orr("KIRC") == 25.0
    assert orr("KIRC", "PD-1+CTLA-4") == 42.0
    assert orr("KIRP") == 28.8
    assert orr("KICH") == 9.5
    renal = [orr("KIRC"), orr("KIRC", "PD-1+CTLA-4"), orr("KIRP"), orr("KICH")]
    assert (min(renal), max(renal)) == (9.5, 42.0)
    # BRCA note: "pembrolizumab 5.0% ... atezolizumab 10.0%"
    assert orr("BRCA_Basal") == 5.0
    assert orr("BRCA_Basal", "PD-L1") == 10.0
    # SARC note: "0% in SARC_LMS and SARC_EWS ... 23% in SARC_UPS ... 25% in
    # SARC_SMARCA4 ... 0% in SARC_GIST"
    assert orr("SARC_LMS") == 0.0 and orr("SARC_EWS") == 0.0
    assert orr("SARC_UPS") == 23.0
    assert orr("SARC_SMARCA4") == 25.0
    assert orr("SARC_GIST") == 0.0
    # STAD_MSI note: "KEYNOTE-059 third-line all-comers (12.0%)"
    assert orr("STAD") == 12.0
