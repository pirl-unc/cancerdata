# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd
import pytest

from oncoref import load_dataset, tumor_references


@pytest.fixture(autouse=True)
def _clear_tumor_reference_caches():
    tumor_references._validated_reference_dataset.cache_clear()
    tumor_references._validated_provenance_dataset.cache_clear()
    yield
    tumor_references._validated_reference_dataset.cache_clear()
    tumor_references._validated_provenance_dataset.cache_clear()


def _tcga_fixture():
    return pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1", "ENSG2", "ENSG1", "ENSG2"],
            "symbol": ["GENE1", "GENE2", "GENE1", "GENE2"],
            "cancer_code": ["LUAD", "LUAD", "BRCA", "BRCA"],
            "tumor_tpm_median": [1.0, 3.0, 4.0, 1.0],
            "tumor_tpm_q1": [0.5, 2.0, 3.0, 0.5],
            "tumor_tpm_q3": [2.0, 4.0, 5.0, 2.0],
            "n_samples": [10, 10, 12, 12],
        }
    )


def _subtype_fixture():
    return pd.DataFrame(
        {
            "symbol": ["GENE1", "GENE2", "GENE1", "GENE2"],
            "cancer_code": ["BEATAML", "BEATAML", "BRCA", "BRCA"],
            "subtype": [
                "BEATAML_ELN_Adverse",
                "BEATAML_ELN_Adverse",
                "BRCA_Her2",
                "BRCA_Her2",
            ],
            "source_cohort": ["BEATAML_OHSU_2022"] * 2 + ["TCGA_BRCA_PAM50"] * 2,
            "tumor_tpm_median": [2.0, 6.0, 3.0, 1.0],
            "tumor_tpm_q1": [1.0, 4.0, 2.0, 0.5],
            "tumor_tpm_q3": [3.0, 8.0, 4.0, 2.0],
            "n_samples": [20, 20, 12, 12],
        }
    )


def test_tcga_accessor_filters_normalizes_and_returns_defensive_copy(monkeypatch):
    source = _tcga_fixture()
    calls = []

    def load_source(name, **kwargs):
        calls.append(name)
        return source

    monkeypatch.setattr(tumor_references, "get_data", load_source)

    first = tumor_references.tcga_deconvolved_expression("LUAD")
    second = tumor_references.tcga_deconvolved_expression("LUAD")

    assert first["tumor_tpm_median"].sum() == pytest.approx(1_000_000)
    assert first.attrs["oncoref"]["derivation_method"] == "tme_deconvolution"
    first.loc[0, "tumor_tpm_median"] = -1
    assert second.loc[0, "tumor_tpm_median"] >= 0
    assert source.loc[0, "tumor_tpm_median"] == 1.0
    assert calls == ["tcga-deconvolved-expression"]


def test_canonical_reference_cache_reuses_owning_frame_and_clears_with_loader(monkeypatch):
    source = _tcga_fixture()
    calls = []

    def load_source(name, **kwargs):
        calls.append(name)
        return source

    monkeypatch.setattr(tumor_references, "get_data", load_source)

    first = tumor_references._validated_reference_dataset("tcga-deconvolved-expression")
    second = tumor_references._validated_reference_dataset("tcga-deconvolved-expression")
    load_dataset._clear_cache()
    third = tumor_references._validated_reference_dataset("tcga-deconvolved-expression")

    assert first is source
    assert second is source
    assert third is source
    assert calls == ["tcga-deconvolved-expression", "tcga-deconvolved-expression"]


def test_subtype_accessor_canonicalizes_legacy_labels_and_keeps_sources_separate(
    monkeypatch,
):
    source = _subtype_fixture()
    monkeypatch.setattr(tumor_references, "get_data", lambda name, **kwargs: source)

    laml = tumor_references.subtype_tumor_reference_expression(
        "LAML",
        subtype_code="LAML_ELNadv",
    )
    brca = tumor_references.subtype_deconvolved_expression(
        "BRCA",
        subtype_code="BRCA_HER2",
    )

    assert set(laml["cancer_code"]) == {"LAML"}
    assert set(laml["subtype"]) == {"LAML_ELNadv"}
    assert set(brca["subtype"]) == {"BRCA_HER2"}
    assert set(brca["source_cohort"]) == {"TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50"}
    assert laml["tumor_tpm_median"].sum() == pytest.approx(1_000_000)
    assert brca["tumor_tpm_median"].sum() == pytest.approx(1_000_000)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("tumor_tpm_q1", -0.1, "negative TPM"),
        ("tumor_tpm_q1", 2.0, "Q1 > median"),
        ("n_samples", 0, "invalid sample counts"),
    ],
)
def test_accessor_rejects_invalid_artifact_values(monkeypatch, column, value, message):
    source = _tcga_fixture()
    source.loc[0, column] = value
    monkeypatch.setattr(tumor_references, "get_data", lambda name, **kwargs: source)

    with pytest.raises(tumor_references.TumorReferenceDataError, match=message):
        tumor_references.tcga_deconvolved_expression(scale="native")


def test_accessor_rejects_duplicate_logical_rows(monkeypatch):
    source = pd.concat([_tcga_fixture(), _tcga_fixture().iloc[[0]]], ignore_index=True)
    monkeypatch.setattr(tumor_references, "get_data", lambda name, **kwargs: source)

    with pytest.raises(tumor_references.TumorReferenceDataError, match="duplicate"):
        tumor_references.tcga_deconvolved_expression(scale="native")


def test_accessor_rejects_unknown_cancer_identity(monkeypatch):
    source = _tcga_fixture()
    source.loc[0, "cancer_code"] = "NOT_A_CANCER"
    monkeypatch.setattr(tumor_references, "get_data", lambda name, **kwargs: source)

    with pytest.raises(tumor_references.TumorReferenceDataError, match="unknown cancer_code"):
        tumor_references.tcga_deconvolved_expression(scale="native")


def test_provenance_validates_methods_and_filters(monkeypatch):
    source = pd.DataFrame(
        {
            "artifact": ["tcga-deconvolved-expression", "subtype-deconvolved-expression"],
            "cancer_code": ["LUAD", "BEATAML"],
            "subtype": [pd.NA, "BEATAML_APL"],
            "source_cohort": ["TCGA_XENA_TOIL_RSEM", "BEATAML_OHSU_2022"],
            "derivation_method": ["tme_deconvolution", "high_purity_passthrough"],
            "derivation_status": [
                "migrated_pinned_artifact",
                "rebuilt_from_oncoref_source_matrix",
            ],
            "source_scale": ["tumor_attributed_tpm", "clean_tpm_16_9_75"],
            "processing_pipeline": ["trufflepig.tcga_decompose", "oncoref importer"],
            "source_artifact": ["tcga.csv.gz", "LAML_APL.parquet"],
            "source_artifact_sha256": ["a" * 64, "b" * 64],
            "source_artifact_commit": ["abc123", pd.NA],
            "source_matrix_version": [pd.NA, "5.22.9"],
            "sample_qc_policy": ["legacy_artifact", "pass"],
            "sample_qc_artifact": [pd.NA, "source-matrix-sample-qc.csv"],
            "sample_qc_artifact_sha256": [pd.NA, "e" * 64],
            "output_artifact_sha256": ["c" * 64, "d" * 64],
            "n_genes": [2, 2],
            "n_reference_samples": [10, 20],
            "notes": ["pinned", "rebuilt"],
        }
    )
    monkeypatch.setattr(tumor_references, "get_data", lambda name, **kwargs: source)

    result = tumor_references.tumor_reference_expression_provenance(
        artifact="subtype-deconvolved-expression.csv.gz",
        cancer_code="LAML",
    )

    assert result[["cancer_code", "subtype", "derivation_method"]].to_dict("records") == [
        {
            "cancer_code": "LAML",
            "subtype": "LAML_APL",
            "derivation_method": "high_purity_passthrough",
        }
    ]


def test_scale_is_an_explicit_mode(monkeypatch):
    monkeypatch.setattr(tumor_references, "get_data", lambda name, **kwargs: _tcga_fixture())

    with pytest.raises(ValueError, match="scale must be one of"):
        tumor_references.tcga_deconvolved_expression(scale="sometimes")


def test_classifier_scale_preserves_quantile_order_when_technical_mass_differs(monkeypatch):
    source = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000210082", "ENSG1"],
            "symbol": ["MT-RNR2", "GENE1"],
            "cancer_code": ["LUAD", "LUAD"],
            "tumor_tpm_median": [100.0, 2.0],
            "tumor_tpm_q1": [1.0, 1.0],
            "tumor_tpm_q3": [200.0, 3.0],
            "n_samples": [10, 10],
        }
    )
    monkeypatch.setattr(tumor_references, "get_data", lambda name, **kwargs: source)

    result = tumor_references.tcga_deconvolved_expression()
    technical = result[result["symbol"].eq("MT-RNR2")].iloc[0]
    biological = result[result["symbol"].eq("GENE1")].iloc[0]

    assert technical[list(tumor_references.TUMOR_REFERENCE_VALUE_COLUMNS)].tolist() == [0, 0, 0]
    assert biological["tumor_tpm_q1"] <= biological["tumor_tpm_median"]
    assert biological["tumor_tpm_median"] <= biological["tumor_tpm_q3"]
    assert result["tumor_tpm_median"].sum() == pytest.approx(1_000_000)


def test_classifier_scale_accepts_compact_categorical_gene_columns(monkeypatch):
    source = _tcga_fixture()
    source["Ensembl_Gene_ID"] = source["Ensembl_Gene_ID"].astype("category")
    source["symbol"] = source["symbol"].astype("category")
    source["cancer_code"] = source["cancer_code"].astype("category")
    monkeypatch.setattr(tumor_references, "get_data", lambda name, **kwargs: source)

    result = tumor_references.tcga_deconvolved_expression("LUAD")

    assert result["tumor_tpm_median"].sum() == pytest.approx(1_000_000)
