# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from oncoref import ici

# PMIDs that the reference audit found pointing to UNRELATED papers (birds, diabetic
# retinopathy, PheWAS methodology, mitochondrial stem cells, etc.). They were corrupted
# in both the ICI and apd1 tables and have been replaced with verified citations; this
# list guards against regression.
CORRUPTED_PMIDS = {
    "PMID:33052747",  # -> NSCLC ctDNA review (was HNSC/KEYNOTE-048)
    "PMID:29260193",  # -> diabetic retinopathy (was STAD/KEYNOTE-059)
    "PMID:27269732",  # -> mitochondrial stem cells (was SCLC/CheckMate-032)
    "PMID:31218020",  # -> evolution of birds (was OV/KEYNOTE-100)
    "PMID:33125908",  # -> trial-design commentary (was MESO/KEYNOTE-158)
    "PMID:34272311",  # -> PheWAS methodology (was KICH/KIRP/KEYNOTE-427)
    "PMID:33812497",  # -> CRC quality-of-life (was BCC/cemiplimab)
    "PMID:31562797",  # -> melanoma 5y (was LUAD combo/CheckMate-227)
    "PMID:32319072",  # -> boron/rat duodenum (was CHOL/KEYNOTE-158)
}


def _pmids(df):
    return {str(v) for v in df["pmid_doi"] if isinstance(v, str) and v.startswith("PMID:")}


def test_no_corrupted_citations_remain():
    ici_pmids = _pmids(ici.cancer_ici_response_df())
    from oncoref import apd1

    apd1_pmids = {
        str(v)
        for v in apd1.cancer_apd1_response_df()["pmid_doi"]
        if isinstance(v, str) and v.startswith("PMID:")
    }
    assert CORRUPTED_PMIDS.isdisjoint(ici_pmids), CORRUPTED_PMIDS & ici_pmids
    assert CORRUPTED_PMIDS.isdisjoint(apd1_pmids), CORRUPTED_PMIDS & apd1_pmids


def test_citation_format_well_formed():
    df = ici.cancer_ici_response_df()
    for v in df["pmid_doi"]:
        if v is None or (isinstance(v, float)):  # blank cells parse as NaN/float
            continue
        s = str(v).strip()
        assert s == "" or s.startswith("PMID:") or s.startswith("DOI:"), s


def test_estimates_table_shape_and_coverage():
    est = ici.cancer_ici_response_estimates_df()
    expected = {
        "cancer_code",
        "regimen",
        "role",
        "drug",
        "trial_name",
        "trial_alias",
        "trial_nct",
        "ref",
        "metric",
        "value",
        "unit",
        "ci_low",
        "ci_high",
        "metric_n",
        "responders",
        "source_verified",
        "value_basis",
    }
    assert expected <= set(est.columns)
    assert len(est) >= 770
    # every (cancer, regimen) in the curated anchor table has >=1 estimate row
    anchor = ici.cancer_ici_response_df()
    anchor_cells = set(zip(anchor["cancer_code"], anchor["regimen"]))
    est_cells = set(zip(est["cancer_code"], est["regimen"]))
    assert anchor_cells <= est_cells, anchor_cells - est_cells
    # ORR is the dominant metric and present for most cells
    assert (est["metric"].str.upper() == "ORR").sum() >= 100


def test_wilson_ci_basics():
    # 0/30 -> lower bound pinned at 0, finite upper bound
    lo, hi = ici._wilson_ci(0, 30)
    assert lo == 0.0 and 0 < hi < 30
    # 50/100 -> interval straddles 50
    lo, hi = ici._wilson_ci(50, 100)
    assert lo < 50 < hi
    # empty -> (None, None)
    assert ici._wilson_ci(0, 0) == (None, None)


def test_pooled_proportion_responder_weighted():
    # NEC_MERKEL anti-PD-L1 has >=2 verified trials reporting responders + n.
    r = ici.pooled_ici_response("NEC_MERKEL", regimen="PD-L1", metric="ORR")
    assert r["poolable"] is True
    assert r["n_studies"] >= 2 and r["n_total"] > 0
    assert r["responders_total"] is not None
    # pooled point estimate must sit inside its own Wilson CI
    assert r["ci_low"] <= r["pooled_pct"] <= r["ci_high"]
    # responder-weighted: pooled == 100 * responders_total / n_total
    assert abs(r["pooled_pct"] - 100 * r["responders_total"] / r["n_total"]) < 0.1
    assert all(ref.startswith(("PMID:", "DOI:")) for ref in r["refs"])


def test_pooled_time_to_event_not_poolable():
    # OS is a median in months -> not responder-poolable.
    r = ici.pooled_ici_response("LIHC", regimen="PD-1+CTLA-4", metric="OS")
    assert r["poolable"] is False
    assert r["pooled_pct"] is None


def test_pooled_verified_only_and_alternates_switches():
    # include_alternates=False never yields more studies than the full pool
    full = ici.pooled_ici_response("BLCA", regimen="PD-1", metric="ORR")
    prim = ici.pooled_ici_response("BLCA", regimen="PD-1", metric="ORR", include_alternates=False)
    assert prim["n_studies"] <= full["n_studies"]
    # verified_only is the default; turning it off cannot drop sources
    loose = ici.pooled_ici_response("READ", regimen="PD-1", metric="ORR", verified_only=False)
    strict = ici.pooled_ici_response("READ", regimen="PD-1", metric="ORR")
    assert len(loose["sources"]) >= len(strict["sources"])


def test_derived_blends_marked_and_never_pooled():
    est = ici.cancer_ici_response_estimates_df()
    derived = est[est["value_basis"] == "derived_blend"]
    # exactly the three all-comer MMR-dependent cells are derived blends
    assert set(zip(derived["cancer_code"], derived["regimen"])) == {
        ("COAD", "PD-1"),
        ("READ", "PD-1"),
        ("UCEC", "PD-1"),
    }
    # the derived all-comer ORR is dropped from pooling even with verified_only=False
    r = ici.pooled_ici_response("COAD", regimen="PD-1", metric="ORR", verified_only=False)
    assert r["sources"] == [] and r["n_studies"] == 0 and r["pooled_pct"] is None


def test_msi_subtype_value_corrected_and_rolls_up():
    anchor = ici.cancer_ici_response_df()

    def orr(code):
        row = anchor[(anchor["cancer_code"] == code) & (anchor["regimen"] == "PD-1")]
        return float(row["orr_pct"].iloc[0])

    # KEYNOTE-177 reports MSI-H/dMMR metastatic colorectal cancer, not separate
    # colon- and rectum-specific MSI rows.
    assert abs(orr("CRC_MSI") - 43.8) < 0.01
    assert "COAD_MSI" not in set(anchor["cancer_code"])
    assert "READ_MSI" not in set(anchor["cancer_code"])
    assert ici.cancer_ici_response("COAD_MSI") == orr("CRC_MSI")
    assert ici.cancer_ici_response("READ_MSI") == orr("CRC_MSI")
    # MSS components present and ~0; all-comer is the (low) prevalence-weighted blend
    assert orr("COAD_MSS") == 0.0
    assert orr("COAD") < orr("CRC_MSI")  # blend is far below the MSI subtype
    # COAD all-comer ~ 43.8 * dMMR-prevalence(~0.13)
    assert 4.0 <= orr("COAD") <= 7.0
    # UCEC corrected to KEYNOTE-158 components (dMMR 48, pMMR 7); all-comer is the
    # roll-up at advanced-EC dMMR prevalence ~20% (0.20*48 + 0.80*7 = 15.2), not 8.
    assert orr("UCEC_POLE") == 100.0
    assert abs(orr("UCEC_MSI") - 48.0) < 0.01
    assert orr("UCEC_CNH") == 7.0 and orr("UCEC_CNL") == 7.0
    rolled = 0.20 * orr("UCEC_MSI") + 0.80 * orr("UCEC_CNH")
    assert abs(orr("UCEC") - rolled) <= 1.0


def test_crc_msi_estimates_are_source_scoped_and_detailed():
    est = ici.cancer_ici_response_estimates_df()
    assert not est["cancer_code"].isin(["COAD_MSI", "READ_MSI"]).any()

    primary = est[
        (est["cancer_code"] == "CRC_MSI") & (est["regimen"] == "PD-1") & (est["role"] == "primary")
    ]
    by_metric = {str(r["metric"]).upper(): r for _, r in primary.iterrows()}
    assert set(by_metric) >= {"ORR", "CRR", "PFS", "DOR"}
    assert float(by_metric["ORR"]["value"]) == 43.8
    assert float(by_metric["ORR"]["ci_low"]) == 35.8
    assert float(by_metric["ORR"]["ci_high"]) == 52.0
    assert int(by_metric["ORR"]["metric_n"]) == 153
    assert int(by_metric["ORR"]["responders"]) == 67
    assert float(by_metric["CRR"]["value"]) == 11.1
    assert int(by_metric["CRR"]["responders"]) == 17
    assert float(by_metric["PFS"]["value"]) == 16.5
    assert float(by_metric["PFS"]["ci_low"]) == 5.4
    assert float(by_metric["PFS"]["ci_high"]) == 32.4

    pooled = ici.pooled_ici_response(
        "COAD_MSI",
        regimen="PD-1",
        metric="ORR",
        verified_only=False,
        include_alternates=False,
    )
    assert pooled["cancer_code"] == "CRC_MSI"
    assert pooled["requested_cancer_code"] == "COAD_MSI"
    assert pooled["pooled_pct"] == 43.8
    assert pooled["n_total"] == 153
    assert pooled["responders_total"] == 67


def test_mss_crc_rows_do_not_carry_msi_h_or_dmmr_contrast_rows():
    est = ici.cancer_ici_response_estimates_df()
    mss = est[est["cancer_code"].isin(["COAD_MSS", "READ_MSS"])]
    text = mss["setting"].fillna("").str.cat(mss["note"].fillna(""), sep=" ")
    assert not text.str.contains(
        r"MSI-H|dMMR|MMR-deficient|mismatch-repair-DEFICIENT",
        case=False,
        regex=True,
    ).any()
    assert not (
        mss["trial_name"].astype(str).str.contains("KEYNOTE-177") & mss["ref"].eq("PMID:33264544")
    ).any()

    crc_msi = est[
        (est["cancer_code"] == "CRC_MSI") & (est["trial_name"] == "KEYNOTE-016 dMMR CRC cohort")
    ]
    by_metric = {str(r["metric"]).upper(): r for _, r in crc_msi.iterrows()}
    assert set(by_metric) == {"ORR", "PFS_RATE", "PFS", "OS"}
    assert float(by_metric["ORR"]["value"]) == 40.0
    assert float(by_metric["ORR"]["ci_low"]) == 12.0
    assert float(by_metric["ORR"]["ci_high"]) == 74.0
    assert int(by_metric["ORR"]["metric_n"]) == 10
    assert int(by_metric["ORR"]["responders"]) == 4
    assert float(by_metric["PFS_RATE"]["value"]) == 78.0
    assert int(by_metric["PFS_RATE"]["metric_n"]) == 9
    assert int(by_metric["PFS_RATE"]["responders"]) == 7


def test_net_nonpancreatic_estimates_are_source_scoped_and_detailed():
    est = ici.cancer_ici_response_estimates_df()
    assert not est["cancer_code"].isin(["NET_LUNG", "NET_MIDGUT", "NET_RECTAL"]).any()

    primary = est[
        (est["cancer_code"] == "NET_NONPANCREATIC")
        & (est["regimen"] == "PD-1+CTLA-4")
        & (est["role"] == "primary")
    ]
    by_metric = {str(r["metric"]).upper(): r for _, r in primary.iterrows()}
    assert set(by_metric) >= {"ORR", "PFS_RATE"}
    assert float(by_metric["ORR"]["value"]) == 0.0
    assert float(by_metric["ORR"]["ci_low"]) == 0.0
    assert float(by_metric["ORR"]["ci_high"]) == 23.0
    assert int(by_metric["ORR"]["metric_n"]) == 14
    assert int(by_metric["ORR"]["responders"]) == 0

    alternate = est[
        (est["cancer_code"] == "NET_NONPANCREATIC")
        & (est["regimen"] == "PD-1+CTLA-4")
        & (est["role"] == "alternate")
    ]
    assert {"ORR", "CBR", "PFS", "PFS_RATE", "OS"} <= set(
        alternate["metric"].astype(str).str.upper()
    )
    assert alternate["setting"].astype(str).str.contains("high-grade").any()

    pooled = ici.pooled_ici_response(
        "NET_LUNG",
        regimen="PD-1+CTLA-4",
        metric="ORR",
        include_alternates=False,
    )
    assert pooled["requested_cancer_code"] == "NET_LUNG"
    assert pooled["cancer_code"] == "NET_NONPANCREATIC"
    assert pooled["pooled_pct"] == 0.0
    assert pooled["n_total"] == 14
    assert pooled["responders_total"] == 0


def test_extrapulmonary_g3_nen_estimates_are_not_lung_lcnec_primary_rows():
    est = ici.cancer_ici_response_estimates_df()
    lung_primary = est[
        (est["cancer_code"] == "NEC_LUNG_LARGECELL")
        & (est["regimen"] == "PD-1")
        & (est["role"] == "primary")
    ]
    assert set(lung_primary["trial_name"]) == {"nivolumab pulmonary LCNEC series"}
    assert set(lung_primary["metric"].str.upper()) == {"ORR"}
    assert not lung_primary["setting"].str.contains("extrapulmonary", case=False).any()

    g3 = est[
        (est["cancer_code"] == "NEN_EXTRAPULMONARY_HG")
        & (est["regimen"] == "PD-1")
        & (est["role"] == "primary")
    ]
    by_metric = {str(r["metric"]).upper(): r for _, r in g3.iterrows()}
    assert {"ORR", "CRR", "DCR", "DOR", "PFS", "OS"} <= set(by_metric)

    orr = by_metric["ORR"]
    assert float(orr["value"]) == 3.4
    assert float(orr["ci_low"]) == 0.1
    assert float(orr["ci_high"]) == 17.8
    assert int(orr["metric_n"]) == 29
    assert int(orr["responders"]) == 1

    dcr = by_metric["DCR"]
    assert float(dcr["value"]) == 24.1
    assert float(dcr["ci_low"]) == 7.9
    assert float(dcr["ci_high"]) == 39.7
    assert int(dcr["responders"]) == 7


def test_dlbc_estimates_do_not_include_pmbcl_keynote170_rows():
    est = ici.cancer_ici_response_estimates_df()

    dlbc = est[(est["cancer_code"] == "DLBC") & (est["regimen"] == "PD-1")]
    assert set(dlbc["trial_name"]) == {"CheckMate 139"}
    assert not dlbc["setting"].str.contains("PMBCL|mediastinal", case=False, regex=True).any()

    pmbcl_primary = est[
        (est["cancer_code"] == "PMBCL")
        & (est["trial_name"] == "KEYNOTE-170")
        & (est["role"] == "primary")
    ]
    assert {"ORR", "CRR", "PR", "DOR", "PFS", "PFS_RATE", "OS", "OS_RATE"} <= set(
        pmbcl_primary["metric"].str.upper()
    )

    primary_analysis = est[
        (est["cancer_code"] == "PMBCL")
        & (est["trial_name"] == "KEYNOTE-170 (2019 primary analysis)")
        & (est["role"] == "alternate")
    ]
    by_metric = {str(r["metric"]).upper(): r for _, r in primary_analysis.iterrows()}
    assert set(by_metric) == {"ORR", "CRR"}

    orr = by_metric["ORR"]
    assert str(orr["ref"]) == "PMID:31609651"
    assert float(orr["value"]) == 45.0
    assert int(orr["metric_n"]) == 53
    assert int(orr["responders"]) == 24

    crr = by_metric["CRR"]
    assert float(crr["value"]) == 13.0
    assert int(crr["responders"]) == 7


def test_luad_stk11_estimates_do_not_include_keynote042_all_comer_nsclc_rows():
    est = ici.cancer_ici_response_estimates_df()

    stk11 = est[(est["cancer_code"] == "LUAD_STK11") & (est["regimen"] == "PD-1")]
    assert "KEYNOTE-042" not in set(stk11["trial_name"])
    assert "PMID:30955977" not in set(stk11["ref"])

    primary = stk11[stk11["role"] == "primary"]
    assert set(primary["trial_name"]) == {"Skoulidis STK11/LKB1 aPD1-resistance analysis"}
    assert set(primary["metric"].str.upper()) == {"ORR"}
    assert set(primary["ref"]) == {"PMID:29773717"}

    alternate_trials = set(stk11[stk11["role"] == "alternate"]["trial_name"])
    assert alternate_trials == {
        "Skoulidis STK11/LKB1 aPD1-resistance analysis",
        "SU2C cohort + CheckMate-057",
    }


def test_keynote427_renal_histology_estimates_are_source_scoped():
    est = ici.cancer_ici_response_estimates_df()
    renal = est[
        (est["ref"].astype(str).str.contains("33529058|4569", regex=True))
        & (est["cancer_code"].isin(["KICH", "KIRP", "RCC_NCC", "RCC_NCC_UNCLASSIFIED"]))
    ]

    kich = renal[renal["cancer_code"] == "KICH"]
    assert set(kich["role"]) == {"primary"}
    assert set(kich["metric"].str.upper()) == {"ORR"}
    assert (
        not kich["setting"]
        .str.contains("overall|papillary|unclassified", case=False, regex=True)
        .any()
    )
    kich_orr = kich.iloc[0]
    assert float(kich_orr["value"]) == 9.5
    assert int(kich_orr["metric_n"]) == 21
    assert int(kich_orr["responders"]) == 2

    kirp = renal[renal["cancer_code"] == "KIRP"]
    assert set(kirp["role"]) == {"primary"}
    assert set(kirp["metric"].str.upper()) == {"ORR"}
    assert (
        not kirp["setting"]
        .str.contains("overall|chromophobe|unclassified", case=False, regex=True)
        .any()
    )
    kirp_orr = kirp.iloc[0]
    assert float(kirp_orr["value"]) == 28.8
    assert int(kirp_orr["metric_n"]) == 118
    assert int(kirp_orr["responders"]) == 34

    ncc = renal[(renal["cancer_code"] == "RCC_NCC") & (renal["role"] == "primary")]
    by_metric = {str(r["metric"]).upper(): r for _, r in ncc.iterrows()}
    assert set(by_metric) == {"ORR", "CRR", "DOR", "PFS", "OS"}
    assert float(by_metric["ORR"]["value"]) == 26.7
    assert int(by_metric["ORR"]["metric_n"]) == 165
    assert int(by_metric["ORR"]["responders"]) == 44

    unclassified = renal[
        (renal["cancer_code"] == "RCC_NCC_UNCLASSIFIED") & (renal["role"] == "primary")
    ].iloc[0]
    assert str(unclassified["metric"]).upper() == "ORR"
    assert float(unclassified["value"]) == 30.8
    assert int(unclassified["metric_n"]) == 26
    assert int(unclassified["responders"]) == 8

    compact = ici.cancer_ici_response_df()
    compact_values = {
        r["cancer_code"]: float(r["orr_pct"])
        for _, r in compact[compact["trial_name"] == "KEYNOTE-427 cohort B"].iterrows()
    }
    assert compact_values["KIRP"] == 28.8
    assert compact_values["RCC_NCC"] == 26.7
    assert compact_values["RCC_NCC_UNCLASSIFIED"] == 30.8

    kich_compact = compact[
        (compact["cancer_code"] == "KICH") & (compact["trial_name"] == "KEYNOTE-427")
    ].iloc[0]
    assert float(kich_compact["orr_pct"]) == 9.5


def test_hnsc_hpv_estimates_are_hpv_stratified_not_all_comer_rows():
    est = ici.cancer_ici_response_estimates_df()
    hpv = est[est["cancer_code"].isin(["HNSC_HPVpos", "HNSC_HPVneg"])]
    bad_setting = hpv["setting"].str.contains(
        "all-comers|biomarker-unselected|PD-L1-positive|PD-L1-negative",
        case=False,
        regex=True,
    )
    assert not bad_setting.any(), hpv[bad_setting][
        ["cancer_code", "trial_name", "metric", "setting"]
    ].to_dict("records")

    pooled = hpv[
        (hpv["trial_name"] == "KEYNOTE-012")
        & (hpv["role"] == "primary")
        & (hpv["metric"].str.upper() == "ORR")
    ]
    by_code = {str(r["cancer_code"]): r for _, r in pooled.iterrows()}
    assert set(by_code) == {"HNSC_HPVpos", "HNSC_HPVneg"}

    pos = by_code["HNSC_HPVpos"]
    assert float(pos["value"]) == 24.0
    assert int(pos["metric_n"]) == 45
    assert int(pos["responders"]) == 11
    assert float(pos["ci_low"]) == 13.0
    assert float(pos["ci_high"]) == 40.0

    neg = by_code["HNSC_HPVneg"]
    assert float(neg["value"]) == 16.0
    assert int(neg["metric_n"]) == 147
    assert int(neg["responders"]) == 23
    assert float(neg["ci_low"]) == 10.0
    assert float(neg["ci_high"]) == 23.0

    expansion = hpv[hpv["trial_name"] == "KEYNOTE-012 expansion cohort"]
    assert set(expansion["metric"].str.upper()) == {"ORR", "PFS_RATE", "OS_RATE"}
    exp_orr = {
        str(r["cancer_code"]): r
        for _, r in expansion[expansion["metric"].str.upper() == "ORR"].iterrows()
    }
    assert float(exp_orr["HNSC_HPVpos"]["value"]) == 32.0
    assert int(exp_orr["HNSC_HPVpos"]["metric_n"]) == 28
    assert int(exp_orr["HNSC_HPVpos"]["responders"]) == 9
    assert float(exp_orr["HNSC_HPVneg"]["value"]) == 14.0
    assert int(exp_orr["HNSC_HPVneg"]["metric_n"]) == 104
    assert int(exp_orr["HNSC_HPVneg"]["responders"]) == 15

    compact = ici.cancer_ici_response_df()
    compact_values = {
        r["cancer_code"]: float(r["orr_pct"])
        for _, r in compact[compact["cancer_code"].isin(["HNSC_HPVpos", "HNSC_HPVneg"])].iterrows()
    }
    assert compact_values == {"HNSC_HPVpos": 24.0, "HNSC_HPVneg": 16.0}

    from oncoref import apd1

    apd1_values = {
        r["cancer_code"]: float(r["apd1_orr_pct"])
        for _, r in apd1.cancer_apd1_response_df()[
            lambda d: d["cancer_code"].isin(["HNSC_HPVpos", "HNSC_HPVneg"])
        ].iterrows()
    }
    assert apd1_values == {"HNSC_HPVpos": 24.0, "HNSC_HPVneg": 16.0}


def test_lusc_checkmate017_orr_and_crr_match_table2():
    est = ici.cancer_ici_response_estimates_df()
    rows = est[
        (est["cancer_code"] == "LUSC")
        & (est["trial_name"] == "CheckMate 017")
        & (est["regimen"] == "PD-1")
        & (est["role"] == "primary")
    ]
    by_metric = {str(r["metric"]).upper(): r for _, r in rows.iterrows()}

    orr = by_metric["ORR"]
    assert float(orr["value"]) == 20.0
    assert int(orr["metric_n"]) == 135
    assert int(orr["responders"]) == 27
    assert float(orr["ci_low"]) == 14.0
    assert float(orr["ci_high"]) == 28.0

    crr = by_metric["CRR"]
    assert int(crr["metric_n"]) == 135
    assert int(crr["responders"]) == 1
    assert float(crr["value"]) == 100 / 135
    assert crr["source_verified"] is True


def test_impower110_estimates_are_nsclc_source_scoped_not_luad_lusc():
    est = ici.cancer_ici_response_estimates_df()
    rows = est[est["trial_name"] == "IMpower110"]
    assert set(rows["cancer_code"]) == {"NSCLC"}
    assert set(rows["metric"].str.upper()) == {"ORR", "DOR", "PFS", "OS"}

    by_metric = {str(r["metric"]).upper(): r for _, r in rows.iterrows()}
    orr = by_metric["ORR"]
    assert float(orr["value"]) == 38.3
    assert int(orr["source_n"]) == 205
    assert int(orr["metric_n"]) == 107
    assert int(orr["responders"]) == 41
    assert "not LUAD- or LUSC-specific" in orr["note"]

    pfs = by_metric["PFS"]
    assert float(pfs["value"]) == 8.1
    assert float(pfs["ci_low"]) == 6.8
    assert float(pfs["ci_high"]) == 11.0
    assert int(pfs["metric_n"]) == 107

    os = by_metric["OS"]
    assert float(os["value"]) == 20.2
    assert float(os["ci_low"]) == 16.5
    assert int(os["metric_n"]) == 107

    compact = ici.cancer_ici_response_df()
    compact_im110 = compact[compact["trial_name"] == "IMpower110"]
    assert compact_im110["cancer_code"].tolist() == ["NSCLC"]
    row = compact_im110.iloc[0]
    assert row["response_denominator"] == 107
    assert row["response_numerator"] == 41
    assert row["histology_match"] == "direct"
    assert row["source_scope"] == "aggregate_source"


def test_trial_columns_split_and_clean():
    import re

    df = ici.cancer_ici_response_df()
    assert {"trial_name", "trial_alias", "trial_nct"} <= set(df.columns)
    assert "trial" not in df.columns
    for _, r in df.iterrows():
        name = str(r["trial_name"]).strip()
        assert name and name.lower() != "nan", f"{r['cancer_code']} missing trial_name"
        assert "(" not in name, f"{r['cancer_code']} trial_name still has parentheses: {name}"
        nct = r["trial_nct"]
        if isinstance(nct, str) and nct.strip():
            assert re.fullmatch(r"NCT\d{8}", nct.strip()), f"{r['cancer_code']} bad NCT {nct}"
        alias = r["trial_alias"]
        if isinstance(alias, str) and alias.strip():
            assert alias.strip() != name, f"{r['cancer_code']} alias echoes trial_name"

    def row(code, regimen):
        m = df[(df["cancer_code"] == code) & (df["regimen"] == regimen)]
        return m.iloc[0]

    # the formerly acronym-less rows are now resolved
    ifct = row("SCLC", "PD-L1")  # IFCT-1603: name IS the protocol code -> no alias
    assert ifct["trial_name"] == "IFCT-1603" and ifct["trial_nct"] == "NCT03059667"
    assert not str(ifct["trial_alias"]).strip() or str(ifct["trial_alias"]) == "nan"
    # formerly pooled/basket PAAD is anchored to KEYNOTE-028 pancreatic cohort
    paad = row("PAAD", "PD-1")
    assert paad["trial_name"] == "KEYNOTE-028"
    assert paad["trial_alias"] == "MK-3475-028"
    assert paad["trial_nct"] == "NCT02054806"


def test_pooled_sources_expose_trial_labels():
    r = ici.pooled_ici_response("NEC_MERKEL", regimen="PD-L1", metric="ORR")
    assert r["sources"], "expected NEC_MERKEL PD-L1 sources"
    s = r["sources"][0]
    assert {"trial_name", "trial_alias", "trial_nct"} <= set(s)


def _num(v):
    import math

    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def test_nbl_response_rows_require_source_denominator():
    # Table 1 gives 11 treated NBL patients; supplementary n=80 is screening only.
    # Every poolable NBL ORR row must carry the treated/evaluable denominator.
    from oncoref import apd1

    est = ici.cancer_ici_response_estimates_df()
    nbl_orr = est[(est["cancer_code"] == "NBL") & (est["metric"].str.upper() == "ORR")]
    missing_n = nbl_orr[nbl_orr["metric_n"].map(_num).isna()]
    assert missing_n.empty

    has_source_denominator = not nbl_orr.empty and nbl_orr["metric_n"].map(_num).notna().all()
    compact_tables = {
        "cancer-ici-response.csv": ici.cancer_ici_response_df(),
        "cancer-apd1-response.csv": apd1.cancer_apd1_response_df(),
    }
    for table_name, df in compact_tables.items():
        if (df["cancer_code"] == "NBL").any():
            assert has_source_denominator, f"{table_name}: NBL row requires source-denominated ORR"


def _truthy(v):
    return str(v).strip().lower() in {"true", "1", "yes"}


def test_estimates_internal_consistency():
    """Machine-checkable invariants on the estimates table — catches transcription / drift
    errors without re-verifying every paper. (Per-paper correctness rests on the audit.)"""
    df = ici.cancer_ici_response_estimates_df()
    HARD_PROPORTION = {"ORR", "CRR", "DCR", "PR"}  # value MUST equal 100·responders/n
    for _, r in df.iterrows():
        m = str(r["metric"]).upper()
        tag = f"{r['cancer_code']}/{r['regimen']}/{m}/{r['role']}"
        assert r["role"] in ("primary", "alternate"), f"{tag}: bad role"
        assert r["value_basis"] in (
            "computed_from_counts",
            "reported",
            "derived_blend",
            "derived_cross_cohort",
            "inferred_from_outcomes",
            "reported_context",
        ), f"{tag}: bad value_basis"
        ref = r["ref"]
        if isinstance(ref, str) and ref.strip():
            assert ref.startswith(("PMID:", "DOI:", "NCT")), f"{tag}: bad ref {ref!r}"
        v, resp, n = _num(r["value"]), _num(r["responders"]), _num(r["metric_n"])
        lo, hi = _num(r["ci_low"]), _num(r["ci_high"])
        if resp is not None and n is not None:
            assert resp <= n, f"{tag}: responders {resp} > n {n}"
        if m in HARD_PROPORTION and v is not None:
            assert 0 <= v <= 100, f"{tag}: proportion value {v} out of range"
            if resp is not None and n:
                assert abs(v - 100 * resp / n) <= 2.0, f"{tag}: value {v} != {resp}/{n}"
        if lo is not None and hi is not None and v is not None:
            assert lo - 0.6 <= v <= hi + 0.6, f"{tag}: value {v} outside CI [{lo},{hi}]"


def test_unverified_rows_do_not_claim_source_verification():
    df = ici.cancer_ici_response_estimates_df()
    reported = df[
        (df["value_basis"].astype(str) == "reported") & (~df["source_verified"].map(_truthy))
    ]
    notes = reported["note"].fillna("").astype(str)
    claims_verified = notes.str.contains(
        r"verified|confirmed|matches full text|confirmed exactly",
        case=False,
        regex=True,
    )
    explicit_uncertainty = notes.str.contains(
        r"unverified|not independently confirmed|not confirmed|not re-confirmed|"
        r"could not (?:be )?confirm(?:ed)?|"
        r"pending primary confirmation|not supported|does not report|not in abstract|"
        r"not captured|not source-verified",
        case=False,
        regex=True,
    )
    bad = reported[claims_verified & ~explicit_uncertainty]
    assert bad.empty, bad[
        ["cancer_code", "regimen", "trial_name", "ref", "metric", "note"]
    ].to_dict("records")


def test_anchor_orr_in_ballpark_of_estimates_primary():
    """The representative anchor (cancer-ici-response.csv) and the estimates table's primary
    ORR for a cell must be in the same ballpark — and there must be exactly ONE primary ORR
    per cell. The anchor is a deliberately *rounded representative* value (per the table
    docstring, "not an exact reproducible constant"), so small gaps vs the precise audited
    value are expected; this only catches GROSS drift (a wrong cell, a 2× transcription
    error) — i.e. the two tables falling out of sync."""
    anchor = ici.cancer_ici_response_df()
    est = ici.cancer_ici_response_estimates_df()
    prim = est[(est["role"] == "primary") & (est["metric"].str.upper() == "ORR")]
    by_cell = {}
    for _, r in prim.iterrows():
        cell = (r["cancer_code"], r["regimen"])
        assert cell not in by_cell, f"{cell}: more than one primary ORR row"
        by_cell[cell] = _num(r["value"])
    for _, a in anchor.iterrows():
        cell = (a["cancer_code"], a["regimen"])
        if cell in by_cell and by_cell[cell] is not None:
            assert abs(float(a["orr_pct"]) - by_cell[cell]) <= 5.0, (
                f"{cell}: anchor {a['orr_pct']} far from estimates primary {by_cell[cell]}"
            )


def test_audited_anchor_values_match_primary_orr():
    anchor = ici.cancer_ici_response_df()
    est = ici.cancer_ici_response_estimates_df()
    audited = {
        ("LIHC", "PD-1"): 20.0,  # CheckMate 040 dose-expansion ORR, PMID:28434648
        ("MDS", "PD-1"): 0.0,  # KEYNOTE-013: no CR/PR by IWG criteria
        ("PAAD", "PD-1"): 0.0,  # KEYNOTE-028 pancreatic cohort: 0/24
        ("SCLC", "PD-1"): 10.0,  # CheckMate 032 nivolumab monotherapy: 10/98
        ("EPN", "PD-1"): 4.5,  # CheckMate 908 pooled EPN arms: 1/22
    }
    for cell, expected in audited.items():
        code, regimen = cell
        a = anchor[(anchor["cancer_code"] == code) & (anchor["regimen"] == regimen)]
        assert len(a) == 1
        assert abs(float(a["orr_pct"].iloc[0]) - expected) < 0.01

        p = est[
            (est["cancer_code"] == code)
            & (est["regimen"] == regimen)
            & (est["role"] == "primary")
            & (est["metric"].str.upper() == "ORR")
        ]
        assert len(p) == 1
        assert abs(float(p["value"].iloc[0]) - expected) < 0.01

    from oncoref import apd1

    apd1_anchor = apd1.cancer_apd1_response_df()
    for (code, regimen), expected in audited.items():
        row = apd1_anchor[
            (apd1_anchor["cancer_code"] == code) & (apd1_anchor["drug_target"] == regimen)
        ]
        assert len(row) == 1
        assert abs(float(row["apd1_orr_pct"].iloc[0]) - expected) < 0.01


def test_paad_keynote028_source_endpoints():
    est = ici.cancer_ici_response_estimates_df()
    rows = est[
        (est["cancer_code"] == "PAAD") & (est["regimen"] == "PD-1") & (est["role"] == "primary")
    ]
    by_metric = {str(r["metric"]): r for _, r in rows.iterrows()}
    assert {"ORR", "PFS", "OS"} <= set(by_metric)

    orr = by_metric["ORR"]
    assert orr["ref"] == "PMID:30557521"
    assert float(orr["value"]) == 0.0
    assert float(orr["ci_low"]) == 0.0 and float(orr["ci_high"]) == 14.2
    assert float(orr["metric_n"]) == 24 and float(orr["responders"]) == 0

    pfs = by_metric["PFS"]
    assert pfs["ref"] == "NCT02054806"
    assert float(pfs["value"]) == 1.7
    assert float(pfs["ci_low"]) == 1.5 and float(pfs["ci_high"]) == 1.8

    os = by_metric["OS"]
    assert os["ref"] == "NCT02054806"
    assert float(os["value"]) == 3.9
    assert float(os["ci_low"]) == 2.8 and float(os["ci_high"]) == 5.5


def test_sclc_checkmate032_source_endpoints():
    est = ici.cancer_ici_response_estimates_df()
    rows = est[(est["cancer_code"] == "SCLC") & (est["ref"] == "PMID:27269741")]

    def row(regimen, role, drug, metric, metric_n=None):
        m = rows[
            (rows["regimen"] == regimen)
            & (rows["role"] == role)
            & (rows["drug"] == drug)
            & (rows["metric"] == metric)
        ]
        if metric_n is not None:
            m = m[m["metric_n"] == metric_n]
        assert len(m) == 1
        return m.iloc[0]

    mono = row("PD-1", "primary", "nivolumab", "ORR")
    assert mono["trial_alias"] == "CA209-032"
    assert mono["trial_nct"] == "NCT01928394"
    assert float(mono["value"]) == 10.0
    assert float(mono["ci_low"]) == 5.0 and float(mono["ci_high"]) == 18.0
    assert float(mono["metric_n"]) == 98 and float(mono["responders"]) == 10

    combo_hi_ipi = row("PD-1", "alternate", "nivolumab + ipilimumab", "ORR", 61)
    assert float(combo_hi_ipi["value"]) == 23.0
    assert float(combo_hi_ipi["ci_low"]) == 13.0 and float(combo_hi_ipi["ci_high"]) == 36.0
    assert float(combo_hi_ipi["metric_n"]) == 61 and float(combo_hi_ipi["responders"]) == 14
    assert bool(combo_hi_ipi["source_verified"]) is True

    combo_hi_nivo = row("PD-1", "alternate", "nivolumab + ipilimumab", "ORR", 54)
    assert float(combo_hi_nivo["value"]) == 19.0
    assert float(combo_hi_nivo["ci_low"]) == 9.0 and float(combo_hi_nivo["ci_high"]) == 31.0
    assert float(combo_hi_nivo["responders"]) == 10
    assert bool(combo_hi_nivo["source_verified"]) is True


def test_epn_checkmate908_pooled_orr_counts():
    est = ici.cancer_ici_response_estimates_df()
    rows = est[
        (est["cancer_code"] == "EPN")
        & (est["regimen"] == "PD-1")
        & (est["ref"] == "PMID:36808285")
        & (est["metric"] == "ORR")
    ]

    primary = rows[rows["role"] == "primary"]
    assert len(primary) == 1
    primary = primary.iloc[0]
    assert primary["drug"] == "nivolumab +/- ipilimumab"
    assert float(primary["source_n"]) == 22
    assert float(primary["metric_n"]) == 22
    assert float(primary["responders"]) == 1
    assert float(primary["value"]) == 4.5
    assert bool(primary["source_verified"]) is True
    assert primary["value_basis"] == "derived_cross_cohort"
    assert primary["ci_basis"] == "computed_wilson"

    combo = rows[rows["role"] == "alternate"]
    assert len(combo) == 1
    combo = combo.iloc[0]
    assert float(combo["metric_n"]) == 10
    assert float(combo["responders"]) == 0
    assert combo["value_basis"] == "reported_context"

    pooled = ici.pooled_ici_response("EPN", regimen="PD-1", metric="ORR", verified_only=False)
    assert pooled["responders_total"] is None
    assert pooled["n_total"] is None
    assert pooled["n_pooled"] == 0
    assert pooled["n_studies"] == 0
    assert pooled["pooled_pct"] is None
    assert pooled["refs"] == []


def test_sarc028_expansion_source_endpoints_and_pools():
    est = ici.cancer_ici_response_estimates_df()
    doi = "DOI:10.1200/JCO.2019.37.15_suppl.11015"

    def row(code, metric, role="primary"):
        m = est[
            (est["cancer_code"] == code)
            & (est["regimen"] == "PD-1")
            & (est["role"] == role)
            & (est["metric"] == metric)
            & (est["ref"] == doi)
        ]
        assert len(m) == 1
        return m.iloc[0]

    ddlps_orr = row("SARC_DDLPS", "ORR")
    assert float(ddlps_orr["source_n"]) == 40
    assert float(ddlps_orr["value"]) == 10.0
    assert float(ddlps_orr["metric_n"]) == 39 and float(ddlps_orr["responders"]) == 4
    assert bool(ddlps_orr["source_verified"]) is True

    ddlps_pfs = row("SARC_DDLPS", "PFS")
    assert float(ddlps_pfs["value"]) == 2.0
    assert float(ddlps_pfs["ci_low"]) == 2.0 and float(ddlps_pfs["ci_high"]) == 4.0
    ddlps_pfs_rate = row("SARC_DDLPS", "PFS_RATE")
    assert float(ddlps_pfs_rate["value"]) == 44.0
    assert float(ddlps_pfs_rate["ci_low"]) == 28.0 and float(ddlps_pfs_rate["ci_high"]) == 60.0
    ddlps_os = row("SARC_DDLPS", "OS")
    assert float(ddlps_os["value"]) == 13.0 and float(ddlps_os["ci_low"]) == 8.0

    ups_orr = row("SARC_UPS", "ORR")
    assert float(ups_orr["source_n"]) == 40
    assert float(ups_orr["value"]) == 23.0
    assert float(ups_orr["metric_n"]) == 40 and float(ups_orr["responders"]) == 9
    ups_crr = row("SARC_UPS", "CRR")
    assert float(ups_crr["value"]) == 5.0
    assert float(ups_crr["metric_n"]) == 40 and float(ups_crr["responders"]) == 2
    ups_pfs = row("SARC_UPS", "PFS")
    assert float(ups_pfs["value"]) == 3.0
    assert float(ups_pfs["ci_low"]) == 2.0 and float(ups_pfs["ci_high"]) == 5.0
    ups_pfs_rate = row("SARC_UPS", "PFS_RATE")
    assert float(ups_pfs_rate["value"]) == 50.0
    assert float(ups_pfs_rate["ci_low"]) == 35.0 and float(ups_pfs_rate["ci_high"]) == 65.0
    ups_os = row("SARC_UPS", "OS")
    assert float(ups_os["value"]) == 12.0
    assert float(ups_os["ci_low"]) == 7.0 and float(ups_os["ci_high"]) == 34.0

    for code, responders, n, pooled in (
        ("SARC_DDLPS", 4, 39, 10.3),
        ("SARC_UPS", 9, 40, 22.5),
    ):
        pooled_orr = ici.pooled_ici_response(
            code,
            regimen="PD-1",
            metric="ORR",
            verified_only=False,
        )
        assert pooled_orr["responders_total"] == responders
        assert pooled_orr["n_total"] == n
        assert pooled_orr["n_pooled"] == 1
        assert pooled_orr["pooled_pct"] == pooled
        assert pooled_orr["refs"] == [doi]
        assert all("context" not in str(s["setting"]).lower() for s in pooled_orr["sources"])
        assert all("comparator" not in str(s["setting"]).lower() for s in pooled_orr["sources"])
        assert all("initial" not in str(s["setting"]).lower() for s in pooled_orr["sources"])

    context = est[
        (est["cancer_code"].isin(["SARC_DDLPS", "SARC_UPS"]))
        & (est["regimen"] == "PD-1")
        & (est["role"] == "alternate")
        & (est["metric"] == "ORR")
    ]
    assert set(context["value_basis"]) == {"reported_context"}


def test_pooled_result_contract():
    r = ici.pooled_ici_response("SKCM", regimen="PD-1", metric="ORR")
    for key in (
        "cancer_code",
        "regimen",
        "metric",
        "poolable",
        "pooled_pct",
        "ci_low",
        "ci_high",
        "n_total",
        "n_studies",
        "n_pooled",
        "refs",
        "value_range",
        "sources",
    ):
        assert key in r
    assert r["cancer_code"] == "SKCM"
    # n_studies is the full evidence count; n_pooled (proportion only) is what entered the
    # responder-weighted pool, so it can only be <= n_studies.
    assert r["n_pooled"] is None or r["n_pooled"] <= r["n_studies"]
