# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-cohort raw per-sample matrix fetch (#35, D-source)."""

import urllib.error

import pandas as pd
import pytest

from oncoref import cancer_types, catalog
from oncoref import source_matrices as sm


def test_registry_and_available_cohorts():
    cohorts = sm.available_cohorts()
    assert len(cohorts) >= 126
    assert "LUAD" in cohorts and "BRCA" in cohorts
    info = sm.cohort_info("LUAD")
    assert info["source_cohort"] and info["n_samples"] > 0


def test_source_id_resolver_distinguishes_physical_and_declared_code_routes():
    beataml = sm.resolution_for_source("beataml-ohsu-2022")
    assert beataml.resolution_method == "declared_cancer_code"
    assert beataml.codes == ("LAML_APL", "LAML_ELNadv", "LAML_ELNfav", "LAML_ELNint")
    assert {matrix.source_cohort for matrix in beataml.matrices} == {"BEATAML_OHSU_2022"}

    tcga_acc = sm.resolution_for_source("tcga-acc")
    assert tcga_acc.resolution_method == "declared_cancer_code"
    assert tcga_acc.codes == ("ACC",)
    assert tcga_acc.matrices[0].source_cohort == "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES"

    mmnst = sm.resolution_for_source("prjna1083972-mmnst")
    assert mmnst.resolution_method == "physical_source"
    assert mmnst.codes == ("SARC_MMNST",)
    assert mmnst.matrices[0].source_cohort == "SRP493407_MMNST_2024"
    assert sm.codes_for_source("prjna1083972-mmnst") == ["SARC_MMNST"]


def test_source_id_resolver_does_not_mix_selected_physical_sources():
    treehouse = sm.resolution_for_source("treehouse-polya-25-01")
    assert treehouse.resolution_method == "physical_source"
    assert "NUTM" not in treehouse.codes
    assert {matrix.source_cohort for matrix in treehouse.matrices} == {"TREEHOUSE_POLYA_25_01"}

    unavailable = sm.resolution_for_source("tcga-sarc")
    assert unavailable.resolution_method == "unavailable"
    assert unavailable.codes == ()
    assert unavailable.availability_reason == "no_selected_matrix_for_declared_cancer_codes"

    with pytest.raises(sm.SourceMatrixError, match="unknown expression source"):
        sm.codes_for_source("not-a-source")


def test_mbl_molecular_subgroups_are_source_matrix_cohorts():
    expected = {
        "MBL_WNT": 17,
        "MBL_SHH": 25,
        "MBL_G3": 44,
        "MBL_G4": 39,
    }

    for code, n_samples in expected.items():
        info = sm.cohort_info(code)
        assert info["source_cohort"] == "TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS"
        assert info["n_samples"] == n_samples
        assert cancer_types.cohort_source_version(code) == "112"


def test_stad_ucec_molecular_subtypes_are_source_matrix_cohorts():
    expected = {
        "STAD_CIN": ("TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE", 221),
        "STAD_MSI": ("TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE", 73),
        "STAD_GS": ("TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE", 50),
        "STAD_EBV": ("TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE", 30),
        "UCEC_CNH": ("TREEHOUSE_POLYA_25_01_TCGA_UCEC_SUBTYPE", 85),
        "UCEC_MSI": ("TREEHOUSE_POLYA_25_01_TCGA_UCEC_SUBTYPE", 41),
        "UCEC_CNL": ("TREEHOUSE_POLYA_25_01_TCGA_UCEC_SUBTYPE", 30),
        "UCEC_POLE": ("TREEHOUSE_POLYA_25_01_TCGA_UCEC_SUBTYPE", 16),
    }

    for code, (source_cohort, n_samples) in expected.items():
        assert code in sm.available_cohorts()
        info = sm.cohort_info(code)
        assert info["source_cohort"] == source_cohort
        assert info["n_samples"] == n_samples


def test_brca_pam50_subtypes_use_split_source_matrix_cohort():
    expected = {
        "BRCA_Basal": 172,
        "BRCA_HER2": 77,
        "BRCA_LumA": 501,
        "BRCA_LumB": 199,
        "BRCA_Normal": 35,
    }

    for code, n_samples in expected.items():
        info = sm.cohort_info(code)
        assert info["source_cohort"] == "TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50"
        assert info["n_samples"] == n_samples


def test_crc_msi_subtypes_use_split_source_matrix_cohort():
    expected = {
        "COAD_MSI": 50,
        "COAD_MSS": 226,
        "READ_MSI": 2,
        "READ_MSS": 83,
    }

    for code, n_samples in expected.items():
        info = sm.cohort_info(code)
        assert info["source_cohort"] == "TREEHOUSE_POLYA_25_01_TCGA_COADREAD_MSI"
        assert info["n_samples"] == n_samples


def test_luad_mutation_subtypes_use_split_source_matrix_cohort():
    expected = {
        "LUAD_EGFR": 67,
        "LUAD_KRAS": 153,
        "LUAD_STK11": 142,
    }

    for code, n_samples in expected.items():
        info = sm.cohort_info(code)
        assert info["source_cohort"] == "TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT"
        assert info["n_samples"] == n_samples


def test_sarc_histology_overlays_use_split_source_matrix_cohort():
    expected = {
        "SARC_WDLPS": 5,
        "SARC_DDLPS": 48,
    }

    for code, n_samples in expected.items():
        info = sm.cohort_info(code)
        assert info["source_cohort"] == "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY"
        assert info["n_samples"] == n_samples

    # PLEOLPS stays on the current four-sample GSE75885 RNA-seq source; the
    # TCGA-SARC overlay has only two samples and should not silently replace it.
    assert sm.cohort_info("SARC_PLEOLPS")["source_cohort"] == "GSE75885_DELESPAUL_2017"


def test_alias_resolves():
    assert sm.local_path("lung_adeno").name == "LUAD.parquet"


def test_treehouse_polya_views_share_physical_sample_namespace():
    root = "TREEHOUSE_POLYA_25_01"

    assert sm.source_sample_namespace(root) == root
    assert sm.source_sample_namespace(f"{root}_TCGA_SAMPLES") == root
    assert sm.source_sample_namespace(f"{root}_TCGA_BRCA_PAM50") == root
    assert sm.source_sample_namespace("TREEHOUSE_RIBOD_25_01") == "TREEHOUSE_RIBOD_25_01"
    assert sm.source_sample_namespace("GSE294016_BARTL_2025_SGC") == ("GSE294016_BARTL_2025_SGC")


def test_unknown_cohort_raises():
    with pytest.raises(sm.SourceMatrixError, match="no per-sample matrix"):
        sm.local_path("NOT_A_COHORT")


def test_release_url_is_per_cohort():
    url = sm.release_url("LUAD")
    assert url.endswith("LUAD_per_sample_tpm.parquet")
    # source matrices are pinned to SOURCE_MATRIX_VERSION (raw inputs), independent of the
    # derived-bundle DATA_VERSION, so a canonical-space bundle bump can't repoint them.
    assert f"source-v{sm.SOURCE_MATRIX_VERSION}" in url


def test_cache_and_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path))
    assert not sm.is_cached("LUAD")

    def fake_urlopen(url):
        import io

        class _R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _R(b"PAR1-fake")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    p = sm.ensure("LUAD")
    assert p.exists() and p.read_bytes() == b"PAR1-fake"
    assert sm.is_cached("LUAD")
    # second ensure is a no-op (already cached)
    assert sm.ensure("LUAD") == p


def test_fetch_download_failure_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path))

    def boom(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(sm.SourceMatrixError, match="failed to download"):
        sm.fetch("BRCA")
    assert not sm.is_cached("BRCA")  # no partial file left


def test_sample_qc_facade_uses_shared_expression_policy(monkeypatch):
    from oncoref import expression

    calls = {}

    def fake_sample_expression_qc(code, **kwargs):
        calls["code"] = code
        calls["kwargs"] = kwargs
        return pd.DataFrame(
            {
                "sample_id": ["S1"],
                "sample_qc_status": ["pass"],
                "source_scale_class": ["linear_rnaseq_tpm"],
            }
        )

    monkeypatch.setattr(expression, "sample_expression_qc", fake_sample_expression_qc)

    out = sm.sample_qc(
        "lung_adeno",
        auto_fetch=False,
        min_detected_genes=123,
        max_zero_fraction=0.7,
    )

    assert calls == {
        "code": "lung_adeno",
        "kwargs": {
            "auto_fetch": False,
            "min_detected_genes": 123,
            "max_zero_fraction": 0.7,
        },
    }
    assert out["sample_qc_status"].tolist() == ["pass"]


def test_sample_qc_manifest_semantic_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    pd.DataFrame(
        {
            "cancer_code": ["LUAD", "LUAD", "BRCA"],
            "sample_id": ["pass_sample", "warn_sample", "fail_sample"],
            "sample_qc_status": ["pass", "warn", "fail"],
            "source_cohort": ["SRC", "SRC", "SRC2"],
        }
    ).to_csv(tmp_path / "source-matrix-sample-qc.csv", index=False)

    out = sm.sample_qc_manifest("lung_adeno", sample_qc="pass_or_warn", auto_fetch=False)

    assert out["sample_id"].tolist() == ["pass_sample", "warn_sample"]
    assert out.attrs["path"].endswith("source-matrix-sample-qc.csv")


# ---- catalog routing + group fetch ----


def test_catalog_per_sample_addressing(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path))
    assert catalog.path("per-sample:LUAD") is None
    called = {}
    monkeypatch.setattr(
        sm, "ensure", lambda code: called.setdefault("code", code) or tmp_path / "x"
    )
    catalog.ensure("per-sample:LUAD")
    assert called["code"] == "LUAD"


def test_catalog_fetch_groups(monkeypatch):
    from oncoref import data_bundle, reference_data

    # hpa group -> only HPA sources downloaded
    monkeypatch.setattr(
        reference_data,
        "local_path",
        lambda n, *a, **k: type("P", (), {"exists": lambda s: False})(),
    )
    hpa_calls = []
    monkeypatch.setattr(reference_data, "download", lambda n, *a, **k: hpa_calls.append(n))
    monkeypatch.setattr(
        data_bundle, "fetch", lambda *a, **k: pytest.fail("bundle must not fetch for hpa group")
    )
    out = catalog.fetch("hpa")
    assert set(out) == set(reference_data.REFERENCE_SOURCES)
    assert set(hpa_calls) == set(reference_data.REFERENCE_SOURCES)


def test_catalog_fetch_source_group(monkeypatch):
    fetched = []
    monkeypatch.setattr(sm, "is_cached", lambda c: False)
    monkeypatch.setattr(sm, "fetch", lambda c, *, force=False: fetched.append(c))
    out = catalog.fetch("source")
    assert len(out) == len(sm.available_cohorts())
    assert out[0].startswith("per-sample:")
