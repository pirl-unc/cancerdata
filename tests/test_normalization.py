# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expression normalization: clean TPM + helpers (#35, Phase N)."""

import inspect

import numpy as np
import pandas as pd
import pytest

import oncoref
from oncoref import gene_families as gf
from oncoref import normalization as norm


def test_public_normalization_functions_are_all_exported():
    # Coherence guard: every undecorated function DEFINED in normalization.py is
    # re-exported from the top-level package, so the surface can't drift back to
    # "clean_tpm exported but its siblings aren't".
    public = [
        name
        for name, obj in vars(norm).items()
        if not name.startswith("_") and inspect.isfunction(obj) and obj.__module__ == norm.__name__
    ]
    missing = [n for n in public if n not in oncoref.__all__ or not hasattr(oncoref, n)]
    assert not missing, f"normalization publics not exported from oncoref: {missing}"


def test_housekeeping_reference_profile_contract():
    profile = norm.housekeeping_reference_profile()

    assert not profile.empty
    assert set(profile.columns) == {
        "Symbol",
        "Ensembl_Gene_ID",
        "reference_tpm",
        "reference_source",
        "profile_version",
    }
    assert set(profile["profile_version"]) == {norm.HOUSEKEEPING_REFERENCE_PROFILE_VERSION}
    assert set(profile["reference_source"]) == {norm.HOUSEKEEPING_REFERENCE_PROFILE_SOURCE}
    assert (profile["reference_tpm"] > 0).all()


def _matrix():
    reference = gf.clean_tpm_censored_reference_tpm()
    tech = [
        gene_id
        for gene_id in sorted(gf.gene_family_ids("mitochondrial"))
        if reference.get(gene_id, 0) > 0
    ][:2]
    gt = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [*tech, "TEST_BIO1", "TEST_BIO2", "TEST_BIO3"],
            "Symbol": ["MT1", "MT2", "BIO1", "BIO2", "BIO3"],
        }
    )
    vals = pd.DataFrame(
        {"s1": [100.0, 200, 300, 400, 500], "s2": [10.0, 0, 100, 200, 300]}, index=gt.index
    )
    return gt, vals


def test_clean_tpm_technical_compartment_budget():
    # _matrix() has 2 technical (mito) + 3 biological genes, no ribosomal -> the technical
    # compartment is pinned to OTHER_TECHNICAL_FRACTION (9%), biology to 75%; the empty ribosomal
    # compartment contributes 0.
    gt, vals = _matrix()
    clean = norm.clean_tpm(vals, gt)
    rem = norm._censored_mask(gt, exclude_ribosomal_proteins=True).to_numpy()
    assert np.allclose(clean.loc[rem].sum(), norm.OTHER_TECHNICAL_FRACTION * 1e6)  # 90k
    assert np.allclose(clean.loc[~rem].sum(), norm.BIOLOGICAL_FRACTION * 1e6)  # 750k
    # Technical composition is fixed to the clean-PolyA reference and therefore
    # identical across samples despite their different raw technical values.
    assert np.allclose(clean.loc[rem, "s1"], clean.loc[rem, "s2"])
    # within-biology ratios preserved (300:400:500)
    bio = clean.loc[~rem, "s1"].to_numpy()
    assert np.allclose(bio / bio.min(), [1.0, 4 / 3, 5 / 3])


def test_clean_tpm_three_compartments():
    # A canonical ribosomal protein gets its OWN 16% budget, distinct from the 9% technical.
    # Pick directly from the category-specific clean-TPM compartment helpers so this
    # tests the public censored-table contract, not broad family membership.
    reference = gf.clean_tpm_censored_reference_tpm()
    rpl = next(
        gene_id for gene_id in sorted(gf.clean_tpm_ribosomal_gene_ids()) if reference[gene_id] > 0
    )
    mito = next(
        gene_id
        for gene_id in sorted(gf.clean_tpm_other_technical_gene_ids())
        if reference[gene_id] > 0
    )
    gt = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [rpl, mito, "TEST_BIO1", "TEST_BIO2"],
            "Symbol": ["RP", "MT", "BIO1", "BIO2"],
        }
    )
    vals = pd.DataFrame({"s1": [5000.0, 5000.0, 300.0, 700.0]}, index=gt.index)
    clean = norm.clean_tpm(vals, gt)
    assert clean.loc[0, "s1"] == pytest.approx(norm.RIBOSOMAL_PROTEIN_FRACTION * 1e6)  # 160k
    assert clean.loc[1, "s1"] == pytest.approx(norm.OTHER_TECHNICAL_FRACTION * 1e6)  # 90k
    assert clean.loc[[2, 3], "s1"].sum() == pytest.approx(norm.BIOLOGICAL_FRACTION * 1e6)  # 750k


def test_clean_tpm_preserves_missing_source_values():
    rpl = sorted(gf.clean_tpm_ribosomal_gene_ids())[0]
    mito = sorted(gf.clean_tpm_other_technical_gene_ids())[0]
    gt = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [rpl, mito, "TEST_BIO1"],
            "Symbol": ["RP", "MT", "BIO"],
        }
    )
    values = pd.DataFrame(
        {
            "missing_ribosomal": [np.nan, 10.0, 20.0],
            "missing_technical": [10.0, np.nan, 20.0],
            "missing_biological": [10.0, 20.0, np.nan],
        },
        index=gt.index,
    )

    clean = norm.clean_tpm(values, gt)

    assert clean.isna().equals(values.isna())


def test_clean_tpm_uses_censored_table_categories_for_ribosomal_budget():
    rpl10ap1 = "ENSG00000244691"
    rpl10l = "ENSG00000165496"
    gt = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [rpl10ap1, rpl10l, "TEST_BIO1"],
            "Symbol": ["RPL10AP1", "RPL10L", "BIO"],
        }
    )
    vals = pd.DataFrame({"s1": [100.0, 200.0, 300.0]}, index=gt.index)

    ribo, tech = norm._compartment_masks(gt, exclude_ribosomal_proteins=True)
    assert ribo.tolist() == [True, False, False]
    assert tech.tolist() == [False, False, False]

    clean = norm.clean_tpm(vals, gt)
    assert clean.loc[0, "s1"] == pytest.approx(norm.RIBOSOMAL_PROTEIN_FRACTION * 1e6)
    assert clean.loc[[1, 2], "s1"].sum() == pytest.approx(norm.BIOLOGICAL_FRACTION * 1e6)
    # RPL10L stays biological despite broad ribosomal-family membership.
    assert clean.loc[1, "s1"] == pytest.approx(300000.0)


def test_clean_tpm_rejects_alternate_compartment_contracts():
    gt, vals = _matrix()
    with pytest.raises(ValueError, match="one canonical 16/9/75 contract"):
        norm.clean_tpm(vals, gt, exclude_ribosomal_proteins=False)
    with pytest.raises(ValueError, match="fraction knobs are deprecated"):
        norm.clean_tpm(vals, gt, other_technical_fraction=0.10)


def test_clean_tpm_compartment_fractions_public_and_sum_to_one():
    import oncoref as cd

    # The applied compartment budgets are public constants (no re-hardcoding the magic
    # numbers). Per-compartment splits + the combined TECHNICAL_FRACTION (matches pirlygenes:
    # the constant is the *combined* 25%, with RIBOSOMAL/OTHER_TECHNICAL the 16/9 split).
    assert (cd.RIBOSOMAL_PROTEIN_FRACTION, cd.OTHER_TECHNICAL_FRACTION) == (0.16, 0.09)
    assert cd.TECHNICAL_FRACTION == 0.25 and cd.BIOLOGICAL_FRACTION == 0.75
    assert cd.RIBOSOMAL_PROTEIN_FRACTION + cd.OTHER_TECHNICAL_FRACTION == cd.TECHNICAL_FRACTION
    assert cd.TECHNICAL_FRACTION + cd.BIOLOGICAL_FRACTION == 1.0


def test_normalize_expression_records_applied_fractions():
    # The clean_tpm path stamps the *applied* compartment budgets into the stats dict, so
    # a consumer reads the values actually used (survives future re-calibration).
    df = pd.DataFrame(
        {"Symbol": ["A", "B"], "Ensembl_Gene_ID": ["E1", "E2"], "s1_TPM": [10.0, 30.0]}
    )
    _, stats = norm.normalize_expression(df, value_cols=["s1_TPM"], censored_fill="fixed_fraction")
    assert stats["ribosomal_protein_fraction"] == norm.RIBOSOMAL_PROTEIN_FRACTION
    assert stats["other_technical_fraction"] == norm.OTHER_TECHNICAL_FRACTION
    assert stats["biological_fraction"] == pytest.approx(0.75)


def test_technical_rna_groups_and_families_public():
    import oncoref as cd
    from oncoref import gene_families, gene_qc

    # Public names (no _-prefixed import across the package boundary) + back-compat aliases.
    assert cd.TECHNICAL_RNA_GROUPS == gene_qc._TECHNICAL_RNA_GROUPS
    assert cd.TECHNICAL_RNA_FAMILIES == gene_families._TECHNICAL_RNA_FAMILIES
    assert "rrna_like" in cd.TECHNICAL_RNA_GROUPS and "rrna" in cd.TECHNICAL_RNA_FAMILIES


def test_clean_tpm_noncensored_ribosomal_paralog_stays_biology():
    # A canonical ribosomal-protein paralog the curated list deliberately keeps OUT of the
    # censored set (testis-restricted RPL10L, a potential antigen) must stay in biology — not
    # get pulled into the 16% ribosomal compartment from the family-file/censored-CSV gap.
    rpl10l = "ENSG00000165496"
    assert rpl10l in gf.gene_family_ids("ribosomal_protein")
    assert rpl10l not in gf.clean_tpm_censored_gene_ids(include_ribosomal_proteins=True)
    gt = pd.DataFrame({"Ensembl_Gene_ID": [rpl10l, "TEST_BIO1"], "Symbol": ["RPL10L", "BIO"]})
    ribo, tech = norm._compartment_masks(gt, exclude_ribosomal_proteins=True)
    assert not ribo.iloc[0] and not tech.iloc[0]  # neither compartment -> biology


def test_clean_tpm_zero_observed_technical_mass_gets_polya_reference_composition():
    gt, vals = _matrix()
    rem = norm._censored_mask(gt, exclude_ribosomal_proteins=True)
    vals.loc[rem] = 0.0  # a sample/library with no technical reads
    clean = norm.clean_tpm(vals, gt)
    assert np.allclose(clean.loc[rem.to_numpy()].sum(), 90_000.0)
    assert np.allclose(clean.loc[~rem.to_numpy()].sum(), 750_000.0)  # biology still 75%


def test_clean_tpm_replaces_ribod_dominant_ncrna_composition_with_polya_profile():
    rn7sl1 = "ENSG00000276168"
    rn7sk = "ENSG00000283293"
    gt = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [rn7sl1, rn7sk, "TEST_BIO"],
            "Symbol": ["RN7SL1", "RN7SK", "BIO"],
        }
    )
    values = pd.DataFrame(
        {
            "ribod_like": [900_000.0, 1.0, 99_999.0],
            "polya_like": [1.0, 900_000.0, 99_999.0],
        }
    )

    clean = norm.clean_tpm(values, gt)
    reference = gf.clean_tpm_censored_reference_tpm()
    expected_ratio = reference[rn7sl1] / reference[rn7sk]

    assert rn7sl1 in gf.technical_rna_gene_ids()
    assert rn7sk in gf.technical_rna_gene_ids()
    assert clean.loc[0, "ribod_like"] / clean.loc[1, "ribod_like"] == pytest.approx(expected_ratio)
    assert np.allclose(clean.loc[[0, 1], "ribod_like"], clean.loc[[0, 1], "polya_like"])
    assert np.allclose(clean.loc[[0, 1]].sum(), norm.OTHER_TECHNICAL_FRACTION * 1e6)


def test_clean_tpm_validates():
    gt, vals = _matrix()
    with pytest.raises(ValueError, match="fraction knobs are deprecated"):
        norm.clean_tpm(vals, gt, other_technical_fraction=1.5)
    with pytest.raises(ValueError, match="fraction knobs are deprecated"):
        norm.clean_tpm(vals, gt, ribosomal_protein_fraction=0.7, other_technical_fraction=0.5)
    with pytest.raises(ValueError, match="gene_table"):
        norm.clean_tpm(vals)


def test_value_cols_excludes_proteoform_members():
    # A proteoform-collapsed frame carries a proteoform_members provenance column; the
    # "everything-not-id" value-column rule must not treat it as a sample (it would
    # crash/poison housekeeping normalization). Uses the shared ID_COLUMNS constant.
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["GA"],
            "proteoform_members": ["GA/GB"],
            "s1": [3.0],
            "s2": [4.0],
        }
    )
    assert norm._value_cols(df) == ["s1", "s2"]


def test_drop_technical_vs_filter_technical_rna():
    gt, _ = _matrix()
    df = gt.assign(s1=1.0)
    # drop_technical_rna removes the censored set (incl. ribosomal); the two
    # mito genes here are technical, so both go.
    assert len(norm.drop_technical_rna(df)) == 3
    assert len(norm.filter_technical_rna(df)) == 3  # mito is technical RNA too


def test_log_and_rank_helpers():
    gt, vals = _matrix()
    df = vals.assign(Ensembl_Gene_ID=gt["Ensembl_Gene_ID"], Symbol=gt["Symbol"])
    log1p = norm.log1p_transform(df)
    assert np.allclose(log1p["s1"], np.log1p(vals["s1"]))
    rank = norm.percentile_rank(df)
    assert rank["s1"].max() <= 100.0 and rank["s1"].min() >= 0.0


def test_normalize_to_housekeeping():
    # housekeeping genes present -> column scaled by the median-of-ratios size factor
    # against the fixed per-gene reference profile.
    hk = ["HK1", "HK2"]
    ref = pd.DataFrame({"Ensembl_Gene_ID": hk, "reference_tpm": [5.0, 10.0]})
    gt = pd.DataFrame({"Ensembl_Gene_ID": [*hk, "ENSG00000999999"], "Symbol": ["H1", "H2", "X"]})
    df = gt.assign(s1=[10.0, 30.0, 100.0])  # HK ratios: 2.0, 3.0 -> size factor 2.5
    out = norm.normalize_to_housekeeping(df, panel_ids=hk, reference_profile=ref)
    assert np.allclose(out["s1"], np.array([10.0, 30.0, 100.0]) / 2.5)


def test_normalize_to_housekeeping_raises_when_no_panel_rows():
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000999998", "ENSG00000999999"],
            "Symbol": ["X1", "X2"],
            "s1": [10.0, 20.0],
        }
    )
    with pytest.raises(ValueError, match="no housekeeping panel genes present"):
        norm.normalize_to_housekeeping(df)

    out = norm.normalize_to_housekeeping(df, errors="nan")
    assert out["s1"].isna().all()


def test_normalize_to_housekeeping_raises_without_id_column():
    df = pd.DataFrame({"Symbol": ["H1", "X"], "s1": [10.0, 20.0]})
    with pytest.raises(ValueError, match="no id column for housekeeping panel"):
        norm.normalize_to_housekeeping(df)

    out = norm.normalize_to_housekeeping(df, errors="ignore")
    assert out.equals(df)


def test_normalize_to_housekeeping_raises_for_mixed_failed_columns():
    hk = ["HK1", "HK2"]
    ref = pd.DataFrame({"Ensembl_Gene_ID": hk, "reference_tpm": [5.0, 15.0]})
    gt = pd.DataFrame({"Ensembl_Gene_ID": [*hk, "ENSG00000999999"], "Symbol": ["H1", "H2", "X"]})
    df = gt.assign(good=[10.0, 30.0, 100.0], failed=[0.0, 0.0, 100.0])

    with pytest.raises(ValueError, match="failed"):
        norm.normalize_to_housekeeping(
            df,
            panel_ids=hk,
            reference_profile=ref,
            drop_zero_panel_values=True,
            min_panel_detected=1,
        )

    out = norm.normalize_to_housekeeping(
        df,
        panel_ids=hk,
        reference_profile=ref,
        drop_zero_panel_values=True,
        min_panel_detected=1,
        errors="nan",
    )
    assert np.allclose(out["good"], np.array([10.0, 30.0, 100.0]) / 2.0)
    assert out["failed"].isna().all()


def test_normalize_to_housekeeping_bad_errors_mode():
    with pytest.raises(ValueError, match="errors must be"):
        norm.normalize_to_housekeeping(pd.DataFrame({"s1": [1.0]}), errors="warn")


# ---- FPKM->TPM / renormalize-to-million ----


def test_renormalize_to_million_rescales_each_column():
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E2"],
            "Symbol": ["A", "B"],
            "s1_TPM": [1.0, 3.0],  # sum 4 -> scale 250000
            "s2_TPM": [10.0, 10.0],  # sum 20 -> scale 50000
        }
    )
    out, stats = norm.renormalize_to_million(df)
    assert out["s1_TPM"].sum() == pytest.approx(1e6)
    assert out["s2_TPM"].sum() == pytest.approx(1e6)
    assert stats["applied"] is True
    assert stats["columns"]["s1_TPM"]["scale"] == pytest.approx(250000.0)
    # id columns untouched
    assert list(out["Symbol"]) == ["A", "B"]


def test_renormalize_ignores_raw_and_zero_columns():
    df = pd.DataFrame(
        {
            "s1_TPM": [1.0, 1.0],
            "s2_TPM_raw": [5.0, 5.0],  # _raw provenance -> never rescaled
            "s3_TPM": [0.0, 0.0],  # zero sum -> left untouched, scale 1.0
        }
    )
    out, stats = norm.renormalize_to_million(df)
    assert list(out["s2_TPM_raw"]) == [5.0, 5.0]  # untouched
    assert "s2_TPM_raw" not in stats["columns"]
    assert stats["columns"]["s3_TPM"]["scale"] == 1.0
    assert out["s1_TPM"].sum() == pytest.approx(1e6)


def test_fpkm_to_tpm_equals_renormalize():
    df = pd.DataFrame({"x_FPKM": [2.0, 6.0, 2.0]})
    out, _ = norm.fpkm_to_tpm(df, value_cols=["x_FPKM"])
    assert out["x_FPKM"].sum() == pytest.approx(1e6)
    assert out["x_FPKM"].to_numpy() == pytest.approx([2e5, 6e5, 2e5])


def test_fpkm_to_tpm_auto_detects_source_style_fpkm_columns():
    df = pd.DataFrame(
        {
            "FPKM_LUAD": [2.0, 8.0],
            "TPM_LUAD": [2.0, 8.0],
            "nTPM_liver": [3.0, 7.0],
        }
    )
    out, stats = norm.fpkm_to_tpm(df)
    assert out["FPKM_LUAD"].to_numpy() == pytest.approx([2e5, 8e5])
    assert out["TPM_LUAD"].to_numpy() == pytest.approx([2.0, 8.0])
    assert out["nTPM_liver"].to_numpy() == pytest.approx([3.0, 7.0])
    assert stats["value_cols"] == ["FPKM_LUAD"]


def test_is_expression_value_col():
    assert norm.is_expression_value_col("LUAD_TPM_clean")
    assert norm.is_expression_value_col("TPM")
    assert norm.is_expression_value_col("FPKM")
    assert norm.is_expression_value_col("FPKM_LUAD")
    assert not norm.is_expression_value_col("TPM_LUAD")
    assert not norm.is_expression_value_col("nTPM_liver")
    assert not norm.is_expression_value_col("LUAD_TPM_raw")
    assert not norm.is_expression_value_col("Ensembl_Gene_ID")


# ---- normalize_expression (technical-RNA zero+renormalize) ----


def test_normalize_expression_zeros_technical_and_preserves_total():
    df = pd.DataFrame(
        {
            "Symbol": ["PRAME", "MT-CO1", "RPL13", "MALAT1", "TP53"],
            "Ensembl_Gene_ID": ["E1", "E2", "E3", "E4", "E5"],
            "s1_TPM": [
                100.0,
                300.0,
                100.0,
                200.0,
                300.0,
            ],  # sum 1000; technical = MT-CO1+MALAT1=500
        }
    )
    out, stats = norm.normalize_expression(df, value_cols=["s1_TPM"])
    by = dict(zip(out["Symbol"], out["s1_TPM"]))
    assert by["MT-CO1"] == 0.0 and by["MALAT1"] == 0.0  # technical zeroed
    assert out["s1_TPM"].sum() == pytest.approx(1000.0)  # total preserved
    # RPL13 is kept (ribosomal protein is NOT technical) -> rescaled up
    assert by["RPL13"] > 100.0
    assert stats["applied"] is True
    assert stats["removed_technical_gene_count"] == 2


def test_normalize_expression_fixed_fraction_delegates_to_clean_tpm():
    df = pd.DataFrame(
        {
            "Symbol": ["A", "B"],
            "Ensembl_Gene_ID": ["E1", "E2"],
            "s1_TPM": [10.0, 30.0],
        }
    )
    out, stats = norm.normalize_expression(
        df, value_cols=["s1_TPM"], censored_fill="fixed_fraction"
    )
    # no technical genes -> biological compartment fills 750k
    assert out["s1_TPM"].sum() == pytest.approx(750000.0, rel=1e-6)
    assert stats["mode"] == "fixed_fraction"


def test_normalize_expression_fixed_fraction_rejects_clean_tpm_knobs():
    df = pd.DataFrame(
        {
            "Symbol": ["A", "B"],
            "Ensembl_Gene_ID": ["E1", "E2"],
            "s1_TPM": [10.0, 30.0],
        }
    )
    with pytest.raises(ValueError, match="fraction knobs are deprecated"):
        norm.normalize_expression(
            df,
            value_cols=["s1_TPM"],
            censored_fill="fixed_fraction",
            other_technical_fraction=0.10,
        )
    with pytest.raises(ValueError, match="one canonical 16/9/75 contract"):
        norm.normalize_expression(
            df,
            value_cols=["s1_TPM"],
            censored_fill="fixed_fraction",
            exclude_ribosomal_proteins=False,
        )


def test_normalize_long_table_groups_independently():
    df = pd.DataFrame(
        {
            "symbol": ["MT-CO1", "TP53", "MT-CO1", "TP53"],
            "Ensembl_Gene_ID": ["E2", "E5", "E2", "E5"],
            "cancer_code": ["LUAD", "LUAD", "SKCM", "SKCM"],
            "tumor_tpm_median": [400.0, 600.0, 100.0, 900.0],
        }
    )
    out, _ = norm.normalize_technical_rna_long_table(
        df, group_cols=["cancer_code"], value_cols=["tumor_tpm_median"]
    )
    by = {(r.cancer_code, r.symbol): r.tumor_tpm_median for r in out.itertuples()}
    assert by[("LUAD", "MT-CO1")] == 0.0
    assert by[("LUAD", "TP53")] == pytest.approx(1000.0)  # 600 rescaled to the 1000 group total
    assert by[("SKCM", "TP53")] == pytest.approx(1000.0)


def test_tpm_to_housekeeping_normalized():
    hk = ["HK1", "HK2"]
    ref = pd.DataFrame({"Ensembl_Gene_ID": hk, "reference_tpm": [50.0, 100.0]})
    df = pd.DataFrame(
        {
            "Symbol": ["HK1", "HK2", "GENE"],
            "Ensembl_Gene_ID": [hk[0], hk[1], "ENSG_X"],
            "s1_TPM": [100.0, 300.0, 50.0],
        }
    )
    out, stats = norm.tpm_to_housekeeping_normalized(
        df, value_cols=["s1_TPM"], panel_ids=hk, reference_profile=ref
    )
    # HK ratios are [2, 3] -> size factor 2.5 -> GENE 50 / 2.5 = 20.
    assert stats["applied"] is True
    assert stats["method"] == "median_of_ratios"
    assert stats["columns"]["s1_TPM"]["denominator"] == pytest.approx(2.5)
    assert out.loc[out["Symbol"] == "GENE", "s1_TPM"].iloc[0] == pytest.approx(20.0)


def test_tpm_to_housekeeping_normalized_handles_duplicate_index_labels():
    hk = ["HK1", "HK2"]
    ref = pd.DataFrame({"Ensembl_Gene_ID": hk, "reference_tpm": [50.0, 100.0]})
    df = pd.DataFrame(
        {
            "Symbol": ["HK1", "HK2", "GENE"],
            "Ensembl_Gene_ID": [hk[0], hk[1], "ENSG_X"],
            "s1_TPM": [100.0, 300.0, 50.0],
        },
        index=[0, 0, 1],
    )

    out, stats = norm.tpm_to_housekeeping_normalized(
        df, value_cols=["s1_TPM"], panel_ids=hk, reference_profile=ref
    )

    assert stats["columns"]["s1_TPM"]["denominator"] == pytest.approx(2.5)
    assert out.loc[out["Symbol"] == "GENE", "s1_TPM"].iloc[0] == pytest.approx(20.0)


def test_tpm_to_housekeeping_normalized_legacy_geomean_method_is_explicit():
    hk = ["HK1", "HK2"]
    df = pd.DataFrame(
        {
            "Symbol": ["HK1", "HK2", "GENE"],
            "Ensembl_Gene_ID": [hk[0], hk[1], "ENSG_X"],
            "s1_TPM": [100.0, 100.0, 50.0],
        }
    )
    with pytest.raises(ValueError, match="legacy_geomean"):
        norm.tpm_to_housekeeping_normalized(
            df, value_cols=["s1_TPM"], panel_ids=hk, method="geomean"
        )

    out, stats = norm.tpm_to_housekeeping_normalized(
        df, value_cols=["s1_TPM"], panel_ids=hk, method="legacy_geomean"
    )
    assert stats["method"] == "legacy_geomean"
    assert stats["columns"]["s1_TPM"]["denominator"] == pytest.approx(100.1, rel=1e-3)
    assert out.loc[out["Symbol"] == "GENE", "s1_TPM"].iloc[0] == pytest.approx(50 / 100.1, rel=1e-3)


def test_tpm_to_housekeeping_normalized_blanks_column_with_no_panel():
    # A column whose housekeeping panel rows are all NaN can't be put on the
    # ratio-to-baseline scale; it must become NaN, not silently stay raw-TPM
    # alongside normalized siblings (the scale-mixing trap).
    hk = ["HK1", "HK2"]
    ref = pd.DataFrame({"Ensembl_Gene_ID": hk, "reference_tpm": [100.0, 100.0]})
    df = pd.DataFrame(
        {
            "Symbol": ["HK1", "HK2", "GENE"],
            "Ensembl_Gene_ID": [hk[0], hk[1], "ENSG_X"],
            "good_TPM": [100.0, 100.0, 50.0],
            "empty_TPM": [np.nan, np.nan, 50.0],  # panel genes unmeasured here
        }
    )
    out, stats = norm.tpm_to_housekeeping_normalized(
        df, value_cols=["good_TPM", "empty_TPM"], panel_ids=hk, reference_profile=ref
    )
    # The measurable column is normalized; the panel-less column is fully NaN, not 50.0.
    assert out.loc[out["Symbol"] == "GENE", "good_TPM"].iloc[0] == pytest.approx(50.0)
    assert out["empty_TPM"].isna().all()
    assert stats["columns"]["empty_TPM"]["denominator"] == 0.0


def test_tpm_to_housekeeping_normalized_gates_sparse_detected_panel():
    hk = ["HK1", "HK2", "HK3"]
    ref = pd.DataFrame({"Ensembl_Gene_ID": hk, "reference_tpm": [100.0, 100.0, 100.0]})
    df = pd.DataFrame(
        {
            "Symbol": ["HK1", "HK2", "HK3", "GENE"],
            "Ensembl_Gene_ID": [*hk, "ENSG_X"],
            "good_TPM": [0.0, 100.0, 100.0, 50.0],
            "sparse_TPM": [0.0, 0.0, 100.0, 50.0],
        }
    )

    with pytest.warns(RuntimeWarning, match="housekeeping normalization skipped"):
        out, stats = norm.tpm_to_housekeeping_normalized(
            df,
            value_cols=["good_TPM", "sparse_TPM"],
            panel_ids=hk,
            panel_name="test_housekeeping",
            reference_profile=ref,
            min_panel_detected=2,
            drop_zero_panel_values=True,
            warn_on_unreliable=True,
        )

    assert out.loc[out["Symbol"] == "GENE", "good_TPM"].iloc[0] == pytest.approx(50.0)
    assert out["sparse_TPM"].isna().all()
    assert stats["columns"]["good_TPM"]["panel_genes_detected"] == 2
    assert stats["columns"]["good_TPM"]["panel_genes_zero"] == 1
    assert stats["columns"]["sparse_TPM"]["panel_genes_detected"] == 1
    assert stats["columns"]["sparse_TPM"]["reason"] == "only 1 nonzero housekeeping panel genes"
