# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-sample curation and molecular-provenance manifests."""

from oncoref import (
    molecular_provenance_for_cancer_code,
    molecular_provenance_for_sample,
    molecular_sample_counts,
    sample_manifest,
    samples_for_cancer_code,
)


def _a_present_code_with_alias():
    # LAML is in the manifest and has the alias 'aml'.
    codes = set(sample_manifest()["cancer_code"].astype(str))
    assert "LAML" in codes
    return "LAML", "aml"


def test_samples_for_cancer_code_resolves_aliases():
    code, alias = _a_present_code_with_alias()
    by_code = samples_for_cancer_code(code)
    by_alias = samples_for_cancer_code(alias)
    assert len(by_code) > 0
    assert by_alias.equals(by_code)  # alias resolves to the same canonical rows


def test_unknown_code_returns_empty_not_error():
    assert samples_for_cancer_code("NOT_A_REAL_CODE").empty


def test_included_only_filters():
    code, _ = _a_present_code_with_alias()
    incl = samples_for_cancer_code(code, included_only=True)
    allrows = samples_for_cancer_code(code, included_only=False)
    assert len(incl) <= len(allrows)


def test_ifs_treehouse_library_and_donor_counts_are_source_specific():
    counts = molecular_sample_counts("SARC_IFS").set_index("source_cohort")

    assert counts.loc["TREEHOUSE_POLYA_25_01", ["n_libraries", "n_donors"]].tolist() == [2, 2]
    assert counts.loc["TREEHOUSE_RIBOD_25_01", ["n_libraries", "n_donors"]].tolist() == [3, 2]


def test_diagnosis_only_samples_never_gain_molecular_confirmation():
    rows = molecular_provenance_for_cancer_code("SARC_IFS")
    diagnosis_only = rows[rows["assay"].eq("diagnosis metadata only")]

    assert not diagnosis_only.empty
    assert set(diagnosis_only["molecular_status"]) == {"unknown"}
    assert diagnosis_only["driver_event"].isna().all()
    assert not diagnosis_only["orthogonal_confirmed"].any()


def test_controlled_molecular_labels_survive_library_and_donor_aggregation():
    rows = molecular_provenance_for_sample("Wegert-CMN-9")

    assert rows["library_id"].tolist() == ["PD37214a", "PD37214c"]
    assert rows["driver_event"].tolist() == ["EGFR kinase-domain ITD"] * 2
    assert rows["orthogonal_confirmed"].all()

    counts = molecular_sample_counts("CMN").set_index("source_cohort")
    ega = counts.loc["EGA_WEGERT_2018_IFS_CMN"]
    assert (ega["n_libraries"], ega["n_donors"], ega["n_expression_libraries"]) == (18, 17, 0)


def test_hcl_manifest_keeps_five_t0_donors_and_one_audited_exclusion():
    rows = samples_for_cancer_code("HCL", included_only=False).set_index("case_id")

    assert set(rows.index) == {"P1", "P2", "P3", "P4", "P5", "P6"}
    assert set(rows.loc[["P2", "P3", "P4", "P5", "P6"], "included"]) == {True}
    assert not bool(rows.loc["P1", "included"])
    assert rows.loc["P1", "exclusion_reason"] == "no_pretreatment_t0_pseudobulk"
    assert set(rows["md5sum"]) == {"4f7676aa9dc44e9aa90e74495b5d9229"}


def test_hcl_molecular_provenance_does_not_infer_braf_status():
    rows = molecular_provenance_for_cancer_code("HCL")

    assert set(rows["donor_id"]) == {"P1", "P2", "P3", "P4", "P5", "P6"}
    assert set(rows["molecular_status"]) == {"unknown"}
    assert rows["driver_event"].isna().all()
    assert not rows["orthogonal_confirmed"].any()
    assert rows["notes"].str.contains("no per-donor BRAF", case=False).all()
