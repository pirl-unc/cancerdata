# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd
import pytest
from scripts.import_tumor_reference_artifacts import (
    BEATAML_SUBTYPES,
    import_artifacts,
    sha256_file,
    summarize_passthrough_matrix,
)


def _matrix(values=((1.0, 3.0), (4.0, 8.0))) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1.2", "ENSG2"],
            "Symbol": ["GENE1", "GENE2"],
            "sample-a": [row[0] for row in values],
            "sample-b": [row[1] for row in values],
        }
    )


def _reference_rows(*, beataml: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["GENE1", "GENE2"],
            "cancer_code": ["BEATAML" if beataml else "BRCA"] * 2,
            "subtype": ["BEATAML_APL" if beataml else "BRCA_Her2"] * 2,
            "source_cohort": ["BEATAML_OHSU_2022" if beataml else "TCGA_BRCA_PAM50"] * 2,
            "tumor_tpm_median": [2.0, 6.0],
            "tumor_tpm_q1": [-1.0 if beataml else 1.0, 4.0],
            "tumor_tpm_q3": [3.0, 8.0],
            "n_samples": [2, 2],
        }
    )


def _tcga_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1", "ENSG2"],
            "symbol": ["GENE1", "GENE2"],
            "cancer_code": ["LUAD", "LUAD"],
            "tumor_tpm_median": [2.0, 6.0],
            "tumor_tpm_q1": [1.0, 4.0],
            "tumor_tpm_q3": [3.0, 8.0],
            "n_samples": [2, 2],
        }
    )


def _sample_qc() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cancer_code": subtype,
                "source_cohort": "BEATAML_OHSU_2022",
                "sample_id": sample,
                "sample_qc_status": status,
            }
            for subtype in BEATAML_SUBTYPES
            for sample, status in (("sample-a", "pass"), ("sample-b", "fail"))
        ]
    )


def test_passthrough_summary_preserves_unversioned_gene_ids_and_quantiles():
    result = summarize_passthrough_matrix(_matrix(), subtype_code="LAML_APL")

    assert result["Ensembl_Gene_ID"].tolist() == ["ENSG1", "ENSG2"]
    assert result["tumor_tpm_median"].tolist() == pytest.approx([2.0, 6.0])
    assert result["tumor_tpm_q1"].tolist() == pytest.approx([1.5, 5.0])
    assert result["tumor_tpm_q3"].tolist() == pytest.approx([2.5, 7.0])
    assert result["n_samples"].tolist() == [2, 2]


def test_passthrough_summary_collapses_duplicate_symbols_without_guessing_an_id():
    matrix = pd.concat([_matrix(), _matrix().iloc[[0]].assign(Ensembl_Gene_ID="ENSG9")])
    result = summarize_passthrough_matrix(matrix, subtype_code="LAML_APL")
    gene1 = result[result["symbol"].eq("GENE1")].iloc[0]

    assert pd.isna(gene1["Ensembl_Gene_ID"])
    assert gene1["tumor_tpm_median"] == pytest.approx(4.0)


def test_import_replaces_only_invalid_beataml_rows_and_records_derivation(tmp_path):
    tcga_path = tmp_path / "input-tcga.csv.gz"
    subtype_path = tmp_path / "input-subtype.csv.gz"
    _tcga_rows().to_csv(tcga_path, index=False)
    pd.concat(
        [_reference_rows(beataml=True), _reference_rows(beataml=False)], ignore_index=True
    ).to_csv(subtype_path, index=False)
    matrices = {}
    for subtype in BEATAML_SUBTYPES:
        path = tmp_path / f"{subtype}.parquet"
        _matrix().to_parquet(path, index=False)
        matrices[subtype] = path
    sample_qc_path = tmp_path / "source-matrix-sample-qc.csv"
    _sample_qc().to_csv(sample_qc_path, index=False)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = import_artifacts(
        tcga_path=tcga_path,
        subtype_path=subtype_path,
        beataml_matrices=matrices,
        sample_qc_path=sample_qc_path,
        output_dir=first_dir,
        source_commit="a" * 40,
    )
    second = import_artifacts(
        tcga_path=tcga_path,
        subtype_path=subtype_path,
        beataml_matrices=matrices,
        sample_qc_path=sample_qc_path,
        output_dir=second_dir,
        source_commit="a" * 40,
    )

    subtype = pd.read_csv(first[1])
    provenance = pd.read_csv(first[2])
    beataml = subtype[subtype["source_cohort"].eq("BEATAML_OHSU_2022")]
    brca = subtype[subtype["cancer_code"].eq("BRCA")]
    assert len(beataml) == 2 * len(BEATAML_SUBTYPES)
    assert set(beataml["n_samples"]) == {1}
    assert (beataml[["tumor_tpm_median", "tumor_tpm_q1", "tumor_tpm_q3"]] >= 0).all().all()
    assert set(brca["source_cohort"]) == {"TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50"}
    assert set(provenance["derivation_method"]) == {
        "tme_deconvolution",
        "high_purity_passthrough",
        "observed_tpm_passthrough",
    }
    assert sha256_file(first[1]) == sha256_file(second[1])
    assert first[0].read_bytes() == tcga_path.read_bytes()


def test_import_rejects_negative_non_beataml_rows(tmp_path):
    tcga_path = tmp_path / "input-tcga.csv.gz"
    subtype_path = tmp_path / "input-subtype.csv.gz"
    _tcga_rows().to_csv(tcga_path, index=False)
    invalid = _reference_rows(beataml=False)
    invalid.loc[0, "tumor_tpm_q1"] = -1.0
    invalid.to_csv(subtype_path, index=False)
    matrices = {}
    for subtype in BEATAML_SUBTYPES:
        path = tmp_path / f"{subtype}.parquet"
        _matrix().to_parquet(path, index=False)
        matrices[subtype] = path

    with pytest.raises(ValueError, match="negative TPM"):
        import_artifacts(
            tcga_path=tcga_path,
            subtype_path=subtype_path,
            beataml_matrices=matrices,
            sample_qc_path=tmp_path / "unused.csv",
            output_dir=tmp_path / "out",
            source_commit="a" * 40,
        )


def test_import_requires_qc_coverage_for_every_matrix_sample(tmp_path):
    tcga_path = tmp_path / "input-tcga.csv.gz"
    subtype_path = tmp_path / "input-subtype.csv.gz"
    sample_qc_path = tmp_path / "source-matrix-sample-qc.csv"
    _tcga_rows().to_csv(tcga_path, index=False)
    _reference_rows(beataml=False).to_csv(subtype_path, index=False)
    sample_qc = _sample_qc()
    sample_qc = sample_qc[~sample_qc["sample_id"].eq("sample-b")]
    sample_qc.to_csv(sample_qc_path, index=False)
    matrices = {}
    for subtype in BEATAML_SUBTYPES:
        path = tmp_path / f"{subtype}.parquet"
        _matrix().to_parquet(path, index=False)
        matrices[subtype] = path

    with pytest.raises(ValueError, match="matrix/QC sample mismatch"):
        import_artifacts(
            tcga_path=tcga_path,
            subtype_path=subtype_path,
            beataml_matrices=matrices,
            sample_qc_path=sample_qc_path,
            output_dir=tmp_path / "out",
            source_commit="a" * 40,
        )
