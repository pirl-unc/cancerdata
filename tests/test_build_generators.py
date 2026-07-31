# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import glob
import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oncoref import expression_builders, expression_source_adapters
from oncoref.cancer_types import cohort_registry

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# Optional real-data parity: the per-sample matrices + pirlygenes' shipped
# percentile artifact live only on a maintainer's machine (~22 GB cache), so this
# is gated like the HPA-dependent tests and skips cleanly everywhere else.
_ACC_MATRIX = glob.glob(
    os.path.expanduser("~/.cache/pirlygenes/expression/*/derived/tcga_acc_per_sample_tpm.parquet")
)
_ACC_REF_LOCATIONS = (
    Path(
        os.path.expanduser(
            "~/code/pirlygenes/pirlygenes/data/cancer-reference-expression-percentiles/ACC.parquet"
        )
    ),
    Path(
        os.path.expanduser(
            "~/.cache/pirlygenes/bundled_data/v5.23.2/"
            "cancer-reference-expression-percentiles/ACC.parquet"
        )
    ),
)
_ACC_REF = next((path for path in _ACC_REF_LOCATIONS if path.exists()), None)
_PARITY_READY = bool(_ACC_MATRIX) and _ACC_REF is not None


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _matrix(genes, samples, values):
    """genes × samples DataFrame with id columns."""
    df = pd.DataFrame(values, columns=samples)
    df.insert(0, "Symbol", [f"G{i}" for i in range(len(genes))])
    df.insert(0, "Ensembl_Gene_ID", genes)
    return df


def _write_gdc_star_counts(path: Path, values: dict[str, float]) -> None:
    rows = [
        {
            "gene_id": "ENSG00000141510.17",
            "gene_name": "TP53",
            "gene_type": "protein_coding",
            "tpm_unstranded": values.get("TP53", 0.0),
        },
        {
            "gene_id": "ENSG00000146648.16",
            "gene_name": "EGFR",
            "gene_type": "protein_coding",
            "tpm_unstranded": values.get("EGFR", 0.0),
        },
        {
            "gene_id": "N_unmapped",
            "gene_name": "N_unmapped",
            "gene_type": "",
            "tpm_unstranded": 999.0,
        },
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _synthetic_sra_salmon_source() -> expression_builders.SraSalmonSource:
    runs = []
    for index in range(6):
        tumor = index < 3
        runs.append(
            expression_builders.SraSalmonRun(
                accession=f"SRR0000000{index}",
                biosample=f"SAMN0000000{index}",
                sample_title=f"{'Tumor' if tumor else 'Normal'}-{index + 1}",
                role="tumor" if tumor else "matched_normal",
                cancer_code="SARC_MMNST" if tumor else None,
                read_urls=(
                    f"https://example.org/SRR0000000{index}_1.fastq.gz",
                    f"https://example.org/SRR0000000{index}_2.fastq.gz",
                ),
                read_md5=("a" * 32, "b" * 32),
            )
        )
    return expression_builders.SraSalmonSource(
        source_id="synthetic-sra-salmon",
        bioproject="PRJNA000000",
        sra_study="SRP000000",
        source_cohort="SYNTHETIC_SRA_SALMON",
        cancer_code="SARC_MMNST",
        runs=tuple(runs),
        reference_fastas=(
            expression_builders.SalmonReferenceFasta(
                url="https://example.org/ensembl.fa.gz",
                file_name="ensembl.fa.gz",
                sha256="c" * 64,
            ),
        ),
        combined_transcriptome_file="ensembl-combined.fa.gz",
        combined_transcriptome_sha256="d" * 64,
        expected_n={"SARC_MMNST": 3},
    )


def _synthetic_sra_salmon_registry_entry() -> dict:
    return {
        "id": "synthetic-sra-salmon",
        "source_type": "sra-salmon",
        "cancer_codes": ["SARC_MMNST"],
        "accession": "PRJNA000000",
        "sra_study": "SRP000000",
        "source_cohort": "SYNTHETIC_SRA_SALMON",
        "expected_samples_by_code": {"SARC_MMNST": 1},
        "reference": {
            "fastas": [
                {
                    "url": "https://example.org/ref.fa.gz",
                    "file_name": "ref.fa.gz",
                    "sha256": "a" * 64,
                }
            ],
            "combined_file": "combined.fa.gz",
            "combined_sha256": "b" * 64,
        },
        "runs": [
            {
                "accession": "SRR00000001",
                "biosample": "SAMN00000001",
                "sample_title": "Tumor-1",
                "role": "tumor",
                "cancer_code": "SARC_MMNST",
                "read_urls": [
                    "https://example.org/read_1.fastq.gz",
                    "https://example.org/read_2.fastq.gz",
                ],
                "read_md5": ["c" * 32, "d" * 32],
            }
        ],
    }


def _write_synthetic_ncbi_assembly_report(tmp_path) -> Path:
    report = tmp_path / "assembly_report.txt"
    rows = [
        [
            "17",
            "assembled-molecule",
            "17",
            "Chromosome",
            "CM000679.2",
            "=",
            "NC_000017.11",
            "Primary Assembly",
            "83257441",
            "chr17",
        ],
        [
            "7",
            "assembled-molecule",
            "7",
            "Chromosome",
            "CM000669.2",
            "=",
            "NC_000007.14",
            "Primary Assembly",
            "159345973",
            "chr7",
        ],
        [
            "KI270706.1",
            "unlocalized-scaffold",
            "1",
            "Chromosome",
            "KI270706.1",
            "=",
            "NT_187361.1",
            "Primary Assembly",
            "175055",
            "chr1_KI270706v1_random",
        ],
        [
            "HG1342_PATCH",
            "fix-patch",
            "1",
            "Chromosome",
            "KQ031383.1",
            "=",
            "NW_012132914.1",
            "PATCHES",
            "467143",
            "chr1_KQ031383v1_fix",
        ],
    ]
    columns = [
        "Sequence-Name",
        "Sequence-Role",
        "Assigned-Molecule",
        "Assigned-Molecule-Location/Type",
        "GenBank-Accn",
        "Relationship",
        "RefSeq-Accn",
        "Assembly-Unit",
        "Sequence-Length",
        "UCSC-style-name",
    ]
    report.write_text(
        "# synthetic assembly report\n"
        + "# "
        + "\t".join(columns)
        + "\n"
        + "\n".join("\t".join(row) for row in rows)
        + "\n"
    )
    return report


# ---------- source-matrix ingestion builders ----------


def test_atomic_write_preserves_existing_artifact_on_failure(tmp_path):
    path = tmp_path / "artifact.csv"
    path.write_text("old\n")

    def _write_then_fail(tmp_path):
        tmp_path.write_text("new\n")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        expression_builders._atomic_write(path, _write_then_fail)

    assert path.read_text() == "old\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_verified_download_resumes_a_partial_file(tmp_path, monkeypatch):
    destination = tmp_path / "reads.fastq.gz"
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.write_bytes(b"abc")
    requests = []

    class _Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 3-5/6"}

        def getcode(self):
            return self.status

    def _urlopen(request, timeout):
        requests.append((request, timeout))
        return _Response(b"def")

    monkeypatch.setattr(expression_builders.urllib.request, "urlopen", _urlopen)
    result = expression_builders.download_verified_file(
        "https://example.org/reads.fastq.gz",
        destination,
        checksum=hashlib.md5(b"abcdef").hexdigest(),
        algorithm="md5",
    )

    assert result.read_bytes() == b"abcdef"
    assert requests[0][0].get_header("Range") == "bytes=3-"
    assert requests[0][1] == 180
    assert not partial.exists()


def test_verified_download_preserves_and_resumes_a_truncated_response(tmp_path, monkeypatch):
    destination = tmp_path / "reads.fastq.gz"
    partial = destination.with_suffix(destination.suffix + ".part")
    responses = [
        (200, {"Content-Length": "6"}, b"abc"),
        (206, {"Content-Range": "bytes 3-5/6"}, b"def"),
    ]
    requests = []

    class _Response(io.BytesIO):
        def __init__(self, status, headers, body):
            super().__init__(body)
            self.status = status
            self.headers = headers

        def getcode(self):
            return self.status

    def _urlopen(request, timeout):
        requests.append((request, timeout))
        return _Response(*responses.pop(0))

    monkeypatch.setattr(expression_builders.urllib.request, "urlopen", _urlopen)
    options = {
        "url": "https://example.org/reads.fastq.gz",
        "destination": destination,
        "checksum": hashlib.md5(b"abcdef").hexdigest(),
        "algorithm": "md5",
    }

    with pytest.raises(RuntimeError, match="incomplete download"):
        expression_builders.download_verified_file(**options)

    assert partial.read_bytes() == b"abc"
    result = expression_builders.download_verified_file(**options)
    assert result.read_bytes() == b"abcdef"
    assert requests[1][0].get_header("Range") == "bytes=3-"


def test_verified_download_discards_a_complete_corrupt_response(tmp_path, monkeypatch):
    destination = tmp_path / "reads.fastq.gz"
    partial = destination.with_suffix(destination.suffix + ".part")

    class _Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": "3"}

        def getcode(self):
            return self.status

    monkeypatch.setattr(
        expression_builders.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(b"bad"),
    )

    with pytest.raises(RuntimeError, match="md5 mismatch"):
        expression_builders.download_verified_file(
            "https://example.org/reads.fastq.gz",
            destination,
            checksum=hashlib.md5(b"xyz").hexdigest(),
            algorithm="md5",
        )

    assert not partial.exists()


def test_geo_matrix_builder_writes_canonical_per_sample_matrix_and_sidecars(tmp_path):
    path = tmp_path / "geo.csv"
    pd.DataFrame(
        {
            "GeneID": ["ENSG00000141510.17", "ENSG00000141510", "ENSG00000146648"],
            "Symbol": ["TP53", "TP53", "EGFR"],
            "annotation": ["a", "b", "c"],
            "sample_1": ["2", "3", "5"],
            "sample_2": ["4", "6", "0"],
        }
    ).to_csv(path, index=False)
    source = expression_builders.GeoMatrixSource(
        cancer_code="X",
        source_cohort="TEST_GEO",
        source_project="GEO",
        file_name=path.name,
        unit="FPKM",
        gene_id_col="GeneID",
        symbol_col="Symbol",
        drop_cols=("annotation",),
        sep=",",
    )

    result = expression_builders.build_source_matrices(
        source,
        cache_dir=tmp_path,
        source_path=path,
    )

    out = pd.read_parquet(result.matrix_paths["X"])
    assert list(out.columns) == ["Ensembl_Gene_ID", "Symbol", "sample_1", "sample_2"]
    assert set(out["Ensembl_Gene_ID"]) == {"ENSG00000141510", "ENSG00000146648"}
    assert np.allclose(out[["sample_1", "sample_2"]].sum(axis=0), [1_000_000.0, 1_000_000.0])
    by_id = out.set_index("Ensembl_Gene_ID")
    assert np.isclose(by_id.loc["ENSG00000141510", "sample_1"], 500_000.0)
    assert np.isclose(by_id.loc["ENSG00000141510", "sample_2"], 1_000_000.0)

    stats = result.mapping_audit["mapping_status"].value_counts().to_dict()
    assert stats == {"resolved": 3}
    literal_zero = result.parse_diagnostics.set_index("value_col").loc["sample_2", "n_literal_zero"]
    assert literal_zero == 1
    assert result.sidecar_paths["mapping_audit"].exists()
    assert result.sidecar_paths["parse_diagnostics"].exists()
    assert result.sidecar_paths["X_sample_qc"].exists()
    assert result.sidecar_paths["summary_rows"].exists()
    assert set(result.sample_qc["sample_id"]) == {"sample_1", "sample_2"}
    assert set(result.sample_qc["source_cohort"]) == {"TEST_GEO"}
    summary = result.summary_rows.set_index("Ensembl_Gene_ID")
    assert list(result.summary_rows.columns) == list(
        expression_builders.REFERENCE_EXPRESSION_COLUMNS
    )
    assert set(summary["cancer_code"]) == {"X"}
    assert set(summary["source_cohort"]) == {"TEST_GEO"}
    assert set(summary["source_project"]) == {"GEO"}
    assert set(summary["tumor_origin"]) == {"primary"}
    assert summary.loc["ENSG00000141510", "n_samples"] == 2
    assert summary.loc["ENSG00000141510", "n_detected"] == 2
    assert summary.loc["ENSG00000141510", "TPM_median"] == pytest.approx(750_000.0)
    assert summary.loc["ENSG00000141510", "TPM_clean_median"] == pytest.approx(562_500.0)
    assert (
        summary.loc["ENSG00000141510", "processing_pipeline"]
        == "test_geo_fpkm_to_tpm_oncoref_canonical_clean_tpm_16_9_75"
    )


def test_canonical_source_builder_preserves_nonstandard_native_unit(tmp_path):
    matrix = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000146648"],
            "Symbol": ["TP53", "EGFR"],
            "sample_1": [600_000.0, 400_000.0],
        }
    )
    source = expression_builders.GeoMatrixSource(
        cancer_code="X",
        source_cohort="TEST_PROXY",
        source_project="GEO",
        file_name="already_canonical.parquet",
        unit="TPM",
        native_unit="TPM proxy",
        source_scale_class="microarray_tpm_proxy",
        linear_tpm_comparable=False,
        tpm_proxy=True,
    )

    result = expression_builders.build_canonical_source_matrices(
        source,
        matrix,
        routed_samples={"X": ["sample_1"]},
        output_dir=tmp_path,
    )

    assert set(result.sample_qc["unit"]) == {"TPM proxy"}
    assert result.summary_rows["source_version"].str.contains("unit=TPM proxy").all()
    assert result.summary_rows["processing_pipeline"].str.contains("tpm_proxy_to_tpm").all()


def test_geo_matrix_builder_reconciles_stale_per_code_artifacts(tmp_path):
    out_dir = tmp_path / "derived"
    out_dir.mkdir()
    stale_matrix = out_dir / "STALE_per_sample_tpm.parquet"
    stale_qc = out_dir / "STALE_sample_qc.csv"
    stale_matrix.write_text("stale matrix")
    stale_qc.write_text("stale qc")
    path = tmp_path / "geo.csv"
    pd.DataFrame(
        {
            "GeneID": ["ENSG00000141510", "ENSG00000146648"],
            "Symbol": ["TP53", "EGFR"],
            "sample_1": ["2", "3"],
        }
    ).to_csv(path, index=False)
    source = expression_builders.GeoMatrixSource(
        cancer_code="LIVE",
        source_cohort="TEST_GEO_STALE",
        source_project="GEO",
        file_name=path.name,
        unit="TPM",
        gene_id_col="GeneID",
        symbol_col="Symbol",
        sep=",",
    )

    result = expression_builders.build_source_matrices(
        source,
        cache_dir=tmp_path,
        source_path=path,
    )

    assert set(result.matrix_paths) == {"LIVE"}
    assert not stale_matrix.exists()
    assert not stale_qc.exists()
    assert (out_dir / "LIVE_per_sample_tpm.parquet").exists()
    assert (out_dir / "LIVE_sample_qc.csv").exists()


def test_geo_matrix_builder_preserves_stale_artifacts_when_no_samples_route(tmp_path):
    out_dir = tmp_path / "derived"
    out_dir.mkdir()
    stale_matrix = out_dir / "STALE_per_sample_tpm.parquet"
    stale_qc = out_dir / "STALE_sample_qc.csv"
    stale_matrix.write_text("stale matrix")
    stale_qc.write_text("stale qc")
    path = tmp_path / "geo.csv"
    pd.DataFrame(
        {
            "GeneID": ["ENSG00000141510"],
            "Symbol": ["TP53"],
            "sample_1": ["2"],
        }
    ).to_csv(path, index=False)
    source = expression_builders.GeoMatrixSource(
        cancer_code=["LIVE"],
        source_cohort="TEST_GEO_EMPTY",
        source_project="GEO",
        file_name=path.name,
        unit="TPM",
        gene_id_col="GeneID",
        symbol_col="Symbol",
        sep=",",
        sample_to_cancer_code=lambda _sample: None,
    )

    with pytest.raises(ValueError, match="no samples were routed"):
        expression_builders.build_source_matrices(
            source,
            cache_dir=tmp_path,
            source_path=path,
        )

    assert stale_matrix.exists()
    assert stale_qc.exists()


def test_geo_matrix_builder_routes_samples_and_reads_transposed_matrix(tmp_path):
    path = tmp_path / "transposed.tsv"
    pd.DataFrame(
        {
            "sample_id": ["tumor_a", "tumor_b"],
            "TP53": ["1", "3"],
            "EGFR": ["1", "1"],
        }
    ).to_csv(path, sep="\t", index=False)
    source = expression_builders.GeoMatrixSource(
        cancer_code=["CODE_A", "CODE_B"],
        source_cohort="TEST_TRANSPOSED",
        file_name=path.name,
        unit="TPM",
        gene_id_col="sample_id",
        transposed=True,
        sample_to_cancer_code=lambda sample: "CODE_A" if sample == "tumor_a" else "CODE_B",
    )

    result = expression_builders.build_source_matrices(
        source,
        cache_dir=tmp_path,
        source_path=path,
    )

    assert set(result.matrix_paths) == {"CODE_A", "CODE_B"}
    code_a = pd.read_parquet(result.matrix_paths["CODE_A"])
    code_b = pd.read_parquet(result.matrix_paths["CODE_B"])
    assert list(code_a.columns) == ["Ensembl_Gene_ID", "Symbol", "tumor_a"]
    assert list(code_b.columns) == ["Ensembl_Gene_ID", "Symbol", "tumor_b"]
    assert set(code_a["Symbol"]) == {"TP53", "EGFR"}
    assert np.isclose(code_a["tumor_a"].sum(), 1_000_000.0)
    assert np.isclose(code_b["tumor_b"].sum(), 1_000_000.0)


def test_transposed_matrix_uses_implicit_sample_index_and_uniquifies_duplicates(tmp_path):
    path = tmp_path / "implicit_sample_ids.tsv"
    path.write_text("TP53\tEGFR\nP-1\t1\t2\nP-1\t3\t4\nP-1.1\t5\t6\n")

    matrix = expression_builders.read_source_expression_matrix(path, transposed=True)

    assert list(matrix.columns) == ["source_row_id", "P-1", "P-1.2", "P-1.1"]
    assert matrix["source_row_id"].tolist() == ["TP53", "EGFR"]
    assert matrix[["P-1", "P-1.2", "P-1.1"]].astype(float).to_numpy().tolist() == [
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 6.0],
    ]


def test_nontransposed_matrix_preserves_implicit_gene_index(tmp_path):
    path = tmp_path / "implicit_gene_ids.tsv"
    path.write_text("sample_a\tsample_b\nENSG1\t1\t2\nENSG2\t3\t4\n")

    matrix = expression_builders.read_source_expression_matrix(path)

    assert matrix.attrs["row_id_col"] == "source_row_id"
    assert list(matrix.columns) == ["source_row_id", "sample_a", "sample_b"]
    assert matrix["source_row_id"].tolist() == ["ENSG1", "ENSG2"]
    assert matrix[["sample_a", "sample_b"]].astype(float).to_numpy().tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]


def test_geo_matrix_builder_routes_from_exact_sample_mapping(tmp_path):
    path = tmp_path / "implicit_sample_ids.tsv"
    path.write_text("TP53\tEGFR\nP-1\t1\t2\nP-1\t3\t4\nP-2\t5\t6\nP-3\t7\t8\n")
    pd.DataFrame(
        {
            "sample_id": ["P-1", "P-1.1", "P-2", "P-3"],
            "cancer_code": ["CODE_A", "CODE_A", "CODE_B", ""],
        }
    ).to_csv(tmp_path / "sample_mapping.csv", index=False)
    source = expression_builders.geo_matrix_source_from_entry(
        {
            "id": "mapped-source",
            "source_type": "geo-matrix",
            "cancer_codes": ["CODE_A", "CODE_B"],
            "source_cohort": "MAPPED_SOURCE",
            "file_name": path.name,
            "unit": "TPM",
            "transposed": True,
            "sample_to_cancer_code": {"mapping_file": "sample_mapping.csv"},
            "expected_samples_by_code": {"CODE_A": 2, "CODE_B": 1},
        },
        data_root=tmp_path,
    )

    result = expression_builders.build_source_matrices(
        source,
        cache_dir=tmp_path,
        source_path=path,
    )

    assert list(result.matrices["CODE_A"].columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "P-1",
        "P-1.1",
    ]
    assert list(result.matrices["CODE_B"].columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "P-2",
    ]
    for code, matrix in result.matrices.items():
        assert {
            source.sample_to_cancer_code(sample)
            for sample in expression_builders.sample_columns(matrix)
        } == {code}
    assert "P-3" not in set(result.sample_qc["sample_id"])


def test_geo_matrix_source_rejects_invalid_declared_routing_contract(tmp_path):
    pd.DataFrame({"sample_id": ["P-1"], "cancer_code": ["UNDECLARED"]}).to_csv(
        tmp_path / "sample_mapping.csv", index=False
    )
    entry = {
        "id": "bad-mapped-source",
        "source_type": "geo-matrix",
        "cancer_codes": ["CODE_A"],
        "source_cohort": "BAD_MAPPED_SOURCE",
        "file_name": "matrix.tsv",
        "unit": "TPM",
        "sample_to_cancer_code": {"mapping_file": "sample_mapping.csv"},
    }

    with pytest.raises(ValueError, match="undeclared cancer codes"):
        expression_builders.geo_matrix_source_from_entry(entry, data_root=tmp_path)

    entry["sample_to_cancer_code"] = None
    entry["expected_samples_by_code"] = {"UNDECLARED": 1}
    with pytest.raises(ValueError, match="expected counts for undeclared"):
        expression_builders.geo_matrix_source_from_entry(entry, data_root=tmp_path)

    entry["expected_samples_by_code"] = {"CODE_A": 1.5}
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        expression_builders.geo_matrix_source_from_entry(entry, data_root=tmp_path)


def test_geo_matrix_builder_rejects_wrong_routed_sample_count_before_writing(tmp_path):
    path = tmp_path / "matrix.tsv"
    pd.DataFrame(
        {
            "sample_id": ["sample_a", "sample_b"],
            "TP53": [1, 3],
            "EGFR": [2, 4],
        }
    ).to_csv(path, sep="\t", index=False)
    output_dir = tmp_path / "derived"
    source = expression_builders.GeoMatrixSource(
        cancer_code="CODE_A",
        source_cohort="COUNTED_SOURCE",
        file_name=path.name,
        unit="TPM",
        gene_id_col="sample_id",
        transposed=True,
        expected_source_samples=2,
        expected_samples_by_code={"CODE_A": 3},
    )

    with pytest.raises(ValueError, match=r"routed CODE_A n=2; expected 3"):
        expression_builders.build_source_matrices(
            source,
            cache_dir=tmp_path,
            output_dir=output_dir,
            source_path=path,
        )

    assert list(output_dir.iterdir()) == []


def test_geo_matrix_builder_rejects_wrong_source_sample_count_before_writing(tmp_path):
    path = tmp_path / "matrix.tsv"
    path.write_text("sample_id\tTP53\nsample_a\t1\nsample_b\t2\n")
    output_dir = tmp_path / "derived"
    source = expression_builders.GeoMatrixSource(
        cancer_code="CODE_A",
        source_cohort="COUNTED_SOURCE",
        file_name=path.name,
        unit="TPM",
        gene_id_col="sample_id",
        transposed=True,
        expected_source_samples=3,
    )

    with pytest.raises(ValueError, match=r"contains 2 source samples; expected 3"):
        expression_builders.build_source_matrices(
            source,
            cache_dir=tmp_path,
            output_dir=output_dir,
            source_path=path,
        )

    assert list(output_dir.iterdir()) == []


def test_geo_matrix_builder_transposed_all_blank_sample_is_missing_not_zero(tmp_path):
    path = tmp_path / "transposed_blank_sample.tsv"
    pd.DataFrame(
        {
            "sample_id": ["tumor_a", "blank_sample", "tumor_b"],
            "TP53": ["1", "", "3"],
            "EGFR": ["1", "", "1"],
        }
    ).to_csv(path, sep="\t", index=False)
    source = expression_builders.GeoMatrixSource(
        cancer_code=["CODE_A", "CODE_B", "CODE_BLANK"],
        source_cohort="TEST_TRANSPOSED_BLANK",
        file_name=path.name,
        unit="TPM",
        gene_id_col="sample_id",
        transposed=True,
        sample_to_cancer_code=lambda sample: {
            "tumor_a": "CODE_A",
            "tumor_b": "CODE_B",
            "blank_sample": "CODE_BLANK",
        }.get(sample),
    )

    result = expression_builders.build_source_matrices(
        source,
        cache_dir=tmp_path,
        source_path=path,
    )

    assert set(result.matrix_paths) == {"CODE_A", "CODE_B"}
    assert "CODE_BLANK" not in result.matrices
    assert "blank_sample" not in set(result.parse_diagnostics["value_col"])
    assert "blank_sample" not in set(result.sample_qc["sample_id"])
    for matrix in result.matrices.values():
        assert "blank_sample" not in matrix.columns
        assert np.allclose(
            matrix[expression_builders.sample_columns(matrix)].sum(axis=0), 1_000_000.0
        )


def test_source_matrix_unit_helpers_validate_raw_counts_lengths():
    df = pd.DataFrame({"gene_id": ["g1", "g2"], "s1": [10.0, 10.0]})
    out = expression_builders.normalize_source_matrix_to_tpm(
        df,
        unit="raw_counts",
        row_id_col="gene_id",
        gene_lengths_kb={"g1": 1.0, "g2": 2.0},
    )

    assert np.isclose(out["s1"].sum(), 1_000_000.0)
    assert np.isclose(out.loc[out["gene_id"] == "g1", "s1"].iloc[0], 666_666.6666666666)
    with pytest.raises(ValueError, match="gene_lengths_kb"):
        expression_builders.normalize_source_matrix_to_tpm(df, unit="raw_counts")


def test_source_matrix_builder_emits_summary_rows_for_raw_counts(tmp_path):
    path = tmp_path / "raw_counts.csv"
    pd.DataFrame(
        {
            "GeneID": ["ENSG00000141510", "ENSG00000146648"],
            "Symbol": ["TP53", "EGFR"],
            "sample_1": [10, 10],
            "sample_2": [30, 10],
        }
    ).to_csv(path, index=False)
    source = expression_builders.GeoMatrixSource(
        cancer_code="RAW",
        source_cohort="TEST_RAW",
        source_project="GEO",
        citation="PMID:1",
        file_name=path.name,
        unit="raw_counts",
        gene_id_col="GeneID",
        symbol_col="Symbol",
        sep=",",
        pipeline_stem="test_raw",
        notes="raw count source notes",
        tumor_origin="metastasis",
        metastasis_site="liver",
    )

    result = expression_builders.build_source_matrices(
        source,
        cache_dir=tmp_path,
        source_path=path,
        gene_lengths_kb={"ENSG00000141510": 1.0, "ENSG00000146648": 2.0},
    )

    summary = result.summary_rows.set_index("Symbol")
    assert set(summary["cancer_code"]) == {"RAW"}
    assert set(summary["notes"]) == {"raw count source notes"}
    assert set(summary["tumor_origin"]) == {"metastasis"}
    assert set(summary["metastasis_site"]) == {"liver"}
    assert set(summary["processing_pipeline"]) == {
        "test_raw_raw_counts_to_tpm_oncoref_canonical_clean_tpm_16_9_75"
    }
    assert summary.loc["TP53", "source_version"].startswith("PMID:1; unit=raw_counts")
    assert summary.loc["TP53", "n_samples"] == 2
    assert summary.loc["TP53", "n_detected"] == 2
    assert summary.loc["TP53", "TPM_median"] > summary.loc["EGFR", "TPM_median"]
    assert pd.read_csv(result.sidecar_paths["summary_rows"]).shape == result.summary_rows.shape


def test_summarize_source_matrix_matches_reference_stat_contract():
    matrix = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000146648"],
            "Symbol": ["TP53", "EGFR"],
            "s0": [0.0, 10.0],
            "s1": [1.0, 10.0],
            "s2": [2.0, 10.0],
            "s3": [3.0, 10.0],
            "s4": [4.0, 10.0],
        }
    )
    source = expression_builders.GeoMatrixSource(
        cancer_code="X",
        source_cohort="TEST_STATS",
        file_name="unused.tsv",
        unit="TPM",
    )

    summary = expression_builders.summarize_source_matrix(
        matrix,
        cancer_code="X",
        source=source,
    ).set_index("Symbol")

    assert summary.loc["TP53", "TPM_median"] == 2.0
    assert summary.loc["TP53", "TPM_mean"] == 2.0
    assert summary.loc["TP53", "TPM_q1"] == 1.0
    assert summary.loc["TP53", "TPM_q3"] == 3.0
    assert summary.loc["TP53", "TPM_min"] == 0.0
    assert summary.loc["TP53", "TPM_max"] == 4.0
    assert summary.loc["TP53", "TPM_std"] == pytest.approx(round(np.sqrt(2.5), 6))
    assert summary.loc["TP53", "TPM_p10"] == 0.4
    assert summary.loc["TP53", "TPM_p90"] == 3.6
    assert summary.loc["TP53", "n_samples"] == 5
    assert summary.loc["TP53", "n_detected"] == 4


def test_summarize_source_matrix_reuses_exact_selected_clean_matrix(monkeypatch):
    matrix = _matrix(["G1"], ["pass", "fail"], np.array([[1.0, 100.0]]))
    selected = matrix[["Ensembl_Gene_ID", "Symbol", "pass"]].copy()
    clean = selected.copy()
    clean["pass"] = 7.0
    source = expression_builders.GeoMatrixSource(
        cancer_code="X",
        source_cohort="TEST",
        file_name="unused.tsv",
        unit="TPM",
    )
    monkeypatch.setattr(
        expression_builders,
        "clean_tpm",
        lambda *args, **kwargs: pytest.fail("selected clean matrix should be reused"),
    )

    summary = expression_builders.summarize_source_matrix(
        selected,
        cancer_code="X",
        source=source,
        clean_matrix=clean,
    )

    assert summary.loc[0, "n_samples"] == 1
    assert summary.loc[0, "TPM_median"] == 1.0
    assert summary.loc[0, "TPM_clean_median"] == 7.0


def test_geo_matrix_source_from_entry_compiles_yaml_filters_and_routing():
    source = expression_builders.geo_matrix_source_from_entry(
        {
            "id": "synthetic",
            "source_type": "geo-matrix",
            "cancer_codes": ["CODE_A", "CODE_B"],
            "source_cohort": "SYNTHETIC",
            "file_url": "https://example.org/source.tsv.gz",
            "file_name": "source.tsv.gz",
            "unit": "log2-TPM",
            "gene_id_col": "",
            "sample_filter": {"include_match": "tumor", "exclude_match": "bad"},
            "sample_to_cancer_code": {
                "rules": [
                    {"match": "^tumor_a", "cancer_code": "CODE_A"},
                    {"match": "^tumor_b", "cancer_code": "CODE_B"},
                ]
            },
            "notes": "source-level summary row notes",
            "pipeline_stem": "synthetic_pipeline",
            "tumor_origin": "metastasis",
            "metastasis_site": "liver",
        }
    )

    assert source.cancer_code == ["CODE_A", "CODE_B"]
    assert source.unit == "log2(TPM+1)"
    assert source.sample_filter(["tumor_a1", "normal_a1", "tumor_bad", "tumor_b1"]) == [
        "tumor_a1",
        "tumor_b1",
    ]
    assert source.sample_to_cancer_code("tumor_a1") == "CODE_A"
    assert source.sample_to_cancer_code("tumor_b1") == "CODE_B"
    assert source.sample_to_cancer_code("normal_a1") is None
    assert source.notes == "source-level summary row notes"
    assert source.pipeline_stem == "synthetic_pipeline"
    assert source.tumor_origin == "metastasis"
    assert source.metastasis_site == "liver"


def test_geo_matrix_source_from_entry_validates_tumor_origin():
    with pytest.raises(ValueError, match="tumor_origin"):
        expression_builders.geo_matrix_source_from_entry(
            {
                "id": "synthetic",
                "source_type": "geo-matrix",
                "cancer_codes": ["CODE_A"],
                "source_cohort": "SYNTHETIC",
                "file_url": "https://example.org/source.tsv.gz",
                "file_name": "source.tsv.gz",
                "unit": "TPM",
                "tumor_origin": "metastatic",
            }
        )


def test_geo_matrix_source_from_registry_loads_packaged_geo_entry():
    source = expression_builders.geo_matrix_source_from_registry("gse328026-sarc-pec")

    assert source.cancer_code == "SARC_PEC"
    assert source.source_cohort == "GSE328026_PECOMA_2026"
    assert source.unit == "TPM"
    assert source.file_name == "GSE328026_TPMs_all_Samples.txt.gz"
    assert source.notes.startswith("PEComa TPM source matrix n=69")
    assert source.pipeline_stem == ""
    assert source.source_version.endswith("harmonized to Ensembl release 112")
    assert (
        source.processing_pipeline
        == "gse328026_pecoma_2026_tpm_to_tpm_ensembl112_clean_tpm_16_9_75"
    )
    assert source.tumor_origin == "primary"
    assert source.metastasis_site is None


def test_gdc_source_from_entry_parses_project_and_filters():
    source = expression_builders.gdc_source_from_entry(
        {
            "id": "synthetic-gdc",
            "source_type": "gdc",
            "cancer_codes": ["CODE_A"],
            "project_id": "TCGA-A;TCGA-B",
            "source_cohort": "SYNTHETIC_GDC",
            "gdc_sample_types": ["Primary Tumor"],
            "gdc_primary_diagnosis_contains": ["synthetic carcinoma"],
            "gdc_sample_id_include_match": "01A$",
            "notes": "synthetic GDC notes",
            "pipeline_stem": "synthetic_gdc",
        }
    )

    assert source.source_id == "synthetic-gdc"
    assert source.project_ids == ("TCGA-A", "TCGA-B")
    assert source.source_cohort == "SYNTHETIC_GDC"
    assert source.cancer_code == "CODE_A"
    assert source.primary_sample_types == ("Primary Tumor",)
    assert source.primary_diagnosis_contains == ("synthetic carcinoma",)
    assert source.sample_id_include_match == "01A$"
    assert source.notes == "synthetic GDC notes"
    assert source.pipeline_stem == "synthetic_gdc"


def test_gdc_source_from_registry_loads_packaged_entry():
    source = expression_builders.gdc_source_from_registry("cgci-blgsp")

    assert source.source_id == "cgci-blgsp"
    assert source.project_ids == ("CGCI-BLGSP",)
    assert source.cancer_code == "BL"
    assert source.source_cohort == "CGCI_BLGSP"


def test_build_gdc_source_matrices_writes_canonical_artifacts(tmp_path):
    file_a = tmp_path / "sample_a.tsv"
    file_b = tmp_path / "sample_b.tsv"
    file_duplicate = tmp_path / "sample_duplicate.tsv"
    file_normal = tmp_path / "normal.tsv"
    _write_gdc_star_counts(file_a, {"TP53": 10.0, "EGFR": 30.0})
    _write_gdc_star_counts(file_b, {"TP53": 20.0, "EGFR": 0.0})
    _write_gdc_star_counts(file_duplicate, {"TP53": 100.0, "EGFR": 100.0})
    _write_gdc_star_counts(file_normal, {"TP53": 1000.0, "EGFR": 1000.0})
    hits = [
        {
            "file_id": "file-a",
            "file_name": file_a.name,
            "analysis": {"workflow_type": "STAR - Counts"},
            "cases": [
                {
                    "submitter_id": "case-a",
                    "project": {"project_id": "TCGA-SYN"},
                    "samples": [{"submitter_id": "sample-a", "sample_type": "Primary Tumor"}],
                    "diagnoses": [{"primary_diagnosis": "Synthetic carcinoma"}],
                }
            ],
        },
        {
            "file_id": "file-dup",
            "file_name": file_duplicate.name,
            "analysis": {"workflow_type": "STAR - Counts"},
            "cases": [
                {
                    "submitter_id": "case-a",
                    "project": {"project_id": "TCGA-SYN"},
                    "samples": [{"submitter_id": "sample-z", "sample_type": "Primary Tumor"}],
                    "diagnoses": [{"primary_diagnosis": "Synthetic carcinoma"}],
                }
            ],
        },
        {
            "file_id": "file-b",
            "file_name": file_b.name,
            "analysis": {"workflow_type": "STAR - Counts"},
            "cases": [
                {
                    "submitter_id": "case-b",
                    "project": {"project_id": "TCGA-SYN"},
                    "samples": [{"submitter_id": "sample-b", "sample_type": "Primary Tumor"}],
                    "diagnoses": [{"primary_diagnosis": "Synthetic carcinoma"}],
                }
            ],
        },
        {
            "file_id": "file-normal",
            "file_name": file_normal.name,
            "analysis": {"workflow_type": "STAR - Counts"},
            "cases": [
                {
                    "submitter_id": "case-normal",
                    "project": {"project_id": "TCGA-SYN"},
                    "samples": [{"submitter_id": "normal", "sample_type": "Solid Tissue Normal"}],
                    "diagnoses": [{"primary_diagnosis": "Synthetic carcinoma"}],
                }
            ],
        },
    ]
    source = expression_builders.GdcSource(
        source_id="synthetic-gdc",
        project_ids=("TCGA-SYN",),
        source_cohort="SYNTHETIC_GDC",
        cancer_code="CODE_A",
        source_project="GDC synthetic",
        primary_sample_types=("Primary Tumor",),
        primary_diagnosis_contains=("Synthetic",),
        sample_lineage_evidence=lambda row: f"synthetic evidence for {row['case_id']}",
        pipeline_stem="synthetic_gdc",
        notes="synthetic GDC source notes",
    )
    manifest = expression_builders.build_gdc_sample_manifest(source, hits)

    result = expression_builders.build_gdc_source_matrices(
        source,
        cache_dir=tmp_path,
        manifest=manifest,
        file_paths={
            "file-a": file_a,
            "file-b": file_b,
            "file-dup": file_duplicate,
            "file-normal": file_normal,
        },
    )

    assert set(result.matrix_paths) == {"CODE_A"}
    out = pd.read_parquet(result.matrix_paths["CODE_A"])
    assert list(out.columns) == ["Ensembl_Gene_ID", "Symbol", "sample-a", "sample-b"]
    assert set(out["Ensembl_Gene_ID"]) == {"ENSG00000141510", "ENSG00000146648"}
    by_symbol = out.set_index("Symbol")
    assert by_symbol.loc["TP53", "sample-a"] == 10.0
    assert by_symbol.loc["EGFR", "sample-b"] == 0.0
    selected_manifest = pd.read_csv(result.sidecar_paths["gdc_sample_manifest"])
    assert selected_manifest["included"].sum() == 2
    assert set(selected_manifest.loc[selected_manifest["included"].astype(bool), "sample_id"]) == {
        "sample-a",
        "sample-b",
    }
    assert "duplicate_sample_for_case_code" in set(selected_manifest["exclusion_reason"])
    assert set(selected_manifest["lineage_evidence_source"]) == {
        "synthetic evidence for case-a",
        "synthetic evidence for case-b",
        "synthetic evidence for case-normal",
    }
    assert result.sidecar_paths["mapping_audit"].exists()
    assert result.sidecar_paths["parse_diagnostics"].exists()
    assert result.sidecar_paths["summary_rows"].exists()
    assert result.sample_qc["sample_id"].tolist() == ["sample-a", "sample-b"]
    summary = result.summary_rows.set_index("Symbol")
    assert set(summary["source_project"]) == {"GDC synthetic"}
    assert set(summary["notes"]) == {"synthetic GDC source notes"}
    assert set(summary["processing_pipeline"]) == {
        "synthetic_gdc_gdc_star_counts_tpm_to_tpm_oncoref_canonical_clean_tpm_16_9_75"
    }
    assert summary.loc["TP53", "TPM_median"] == 15.0
    assert summary.loc["TP53", "n_samples"] == 2


def test_recount3_source_from_registry_loads_packaged_routes():
    source = expression_builders.recount3_source_from_registry("gse98894-midnet")

    assert source.source_id == "gse98894-midnet"
    assert source.srp == "SRP107025"
    assert source.cancer_code == ["NET_MIDGUT", "NET_PANCREAS", "NET_RECTAL"]
    assert source.expected_n == {"NET_MIDGUT": 81, "NET_PANCREAS": 113, "NET_RECTAL": 18}
    assert source.tumor_origin == "primary"
    assert source.sample_to_cancer_code({"origin": "ileum"}, "") == "NET_MIDGUT"
    assert source.sample_to_cancer_code({"origin": "pancreas"}, "") == "NET_PANCREAS"
    assert source.sample_to_cancer_code({"origin": "rectal"}, "") == "NET_RECTAL"
    assert source.sample_to_cancer_code({"origin": "lung"}, "") is None


def test_sra_ncbi_count_source_from_registry_loads_complete_mmnst_manifest():
    source = expression_builders.sra_ncbi_count_source_from_registry("prjna1083972-mmnst")

    assert source.bioproject == "PRJNA1083972"
    assert source.sra_study == "SRP493407"
    assert source.cancer_code == "SARC_MMNST"
    assert source.expected_n == {"SARC_MMNST": 3}
    assert source.annotation_release == "GCF_000001405.40-RS_2025_08"
    assert source.annotation_sha256 == (
        "4920f0eae7e2197c50b67a201e06d657387137b49dd60f474b4f1d5b29334051"
    )
    assert source.assembly_report_sha256 == (
        "64318ddff470b69b261a667d813210044f60d4ce654253a547db80ff73638d38"
    )
    assert len(source.runs) == 6
    assert [run.accession for run in source.runs if run.included_in_reference] == [
        "SRR28227826",
        "SRR28227825",
        "SRR28227824",
    ]
    assert {run.role for run in source.runs if not run.included_in_reference} == {"normal_control"}

    manifest = expression_builders.sra_ncbi_count_run_manifest(source)
    assert manifest["included"].value_counts().to_dict() == {True: 3, False: 3}
    excluded = manifest.loc[~manifest["included"]]
    assert set(excluded["cancer_code"]) == {""}
    assert set(excluded["exclusion_reason"]) == {
        "independent_normal_control_excluded_from_tumor_reference"
    }
    assert set(manifest["analysis_accession"]) == {
        "SRZ3595337",
        "SRZ3709728",
        "SRZ3615119",
        "SRZ3647871",
        "SRZ3665545",
        "SRZ3633285",
    }


def test_sra_ncbi_count_source_rejects_a_control_routed_to_tumor():
    entry = deepcopy(expression_builders.sra_ncbi_count_source_entries()[0])
    control = next(run for run in entry["runs"] if run["role"] == "normal_control")
    control["cancer_code"] = "SARC_MMNST"

    with pytest.raises(ValueError, match=r"normal-control SRA run.*must not declare"):
        expression_builders.sra_ncbi_count_source_from_entry(entry)


def test_sra_salmon_source_rejects_a_control_routed_to_tumor():
    entry = {
        "id": "bad-sra",
        "source_type": "sra-salmon",
        "cancer_codes": ["SARC_MMNST"],
        "accession": "PRJNA000000",
        "sra_study": "SRP000000",
        "source_cohort": "BAD_SRA",
        "expected_samples_by_code": {"SARC_MMNST": 0},
        "reference": {
            "fastas": [
                {
                    "url": "https://example.org/ref.fa.gz",
                    "file_name": "ref.fa.gz",
                    "sha256": "a" * 64,
                }
            ],
            "combined_file": "combined.fa.gz",
            "combined_sha256": "d" * 64,
        },
        "runs": [
            {
                "accession": "SRR00000001",
                "biosample": "SAMN00000001",
                "sample_title": "Normal-1",
                "role": "matched_normal",
                "cancer_code": "SARC_MMNST",
                "read_urls": [
                    "https://example.org/read_1.fastq.gz",
                    "https://example.org/read_2.fastq.gz",
                ],
                "read_md5": ["b" * 32, "c" * 32],
            }
        ],
    }

    with pytest.raises(ValueError, match="must not declare a cancer_code"):
        expression_builders.sra_salmon_source_from_entry(entry)


@pytest.mark.parametrize(
    "override",
    [
        "--output=/tmp/not-the-build-cache",
        "-i/path/to/index",
        "-lA",
        "-1reads_1.fastq.gz",
        "-2reads_2.fastq.gz",
        "-o/tmp/not-the-build-cache",
        "-p8",
    ],
)
def test_sra_salmon_source_rejects_builder_option_overrides(override):
    entry = _synthetic_sra_salmon_registry_entry()
    entry["salmon_args"] = ["--validateMappings", override]

    with pytest.raises(ValueError, match="override builder-owned options"):
        expression_builders.sra_salmon_source_from_entry(entry)


@pytest.mark.parametrize("file_kind", ["component", "combined"])
def test_sra_salmon_source_requires_gzip_fasta_file_names(file_kind):
    entry = deepcopy(_synthetic_sra_salmon_registry_entry())
    if file_kind == "component":
        entry["reference"]["fastas"][0]["file_name"] = "reference.fa"
    else:
        entry["reference"]["combined_file"] = "combined.fa"

    with pytest.raises(ValueError, match=r"must be gzip-compressed with a \.gz file name"):
        expression_builders.sra_salmon_source_from_entry(entry)


def test_treehouse_source_from_registry_loads_direct_cohort_routes():
    source = expression_builders.treehouse_source_from_registry("treehouse-polya-25-01")

    assert source.source_id == "treehouse-polya-25-01"
    assert source.source_cohort == "TREEHOUSE_POLYA_25_01"
    assert source.tpm_file.startswith("Tumor-25.01-Polya")
    assert source.clinical_file.startswith("clinical_Treehouse")
    assert len(source.cohorts) == 26
    by_code = {cohort.cancer_code: cohort for cohort in source.cohorts}
    assert by_code["SARC_EWS"].disease_label == "Ewing sarcoma"
    assert by_code["SARC_MPNST"].group == "sarc_rare_direct"
    assert by_code["SARC_GIST"].group == "sarc_subtypes"

    rare = expression_builders.treehouse_cohorts_for_group("sarc_rare_direct")
    assert "SARC_MPNST" in {cohort.cancer_code for cohort in rare}

    ribod = expression_builders.treehouse_source_from_registry("treehouse-ribod-25-01")
    assert [cohort.cancer_code for cohort in ribod.cohorts] == ["SARC_CHOR", "RB"]


def _synthetic_mbl_matrix():
    marker_ids = expression_builders.MEDULLOBLASTOMA_SUBGROUP_MARKER_GENE_IDS
    return pd.DataFrame(
        {
            "Ensembl_Gene_ID": [*marker_ids.values(), "ENSG_OTHER"],
            "Symbol": ["WIF1", "GLI2", "MYC", "KCNA1", "OTHER"],
            "wnt_sample": [10.0, 2.0, 3.0, 4.0, 1.0],
            "shh_sample": [1.0, 10.0, 3.0, 4.0, 1.0],
            "g3_sample": [1.0, 2.0, 10.0, 4.0, 1.0],
            "g4_sample": [1.0, 2.0, 3.0, 10.0, 1.0],
        }
    )


def test_medulloblastoma_subgroup_matrices_use_one_explicit_marker_winner():
    groups = expression_builders.medulloblastoma_subgroup_sample_ids(_synthetic_mbl_matrix())
    matrices = expression_builders.medulloblastoma_subgroup_matrices(_synthetic_mbl_matrix())

    assert groups == {
        "MBL_WNT": ["wnt_sample"],
        "MBL_SHH": ["shh_sample"],
        "MBL_G3": ["g3_sample"],
        "MBL_G4": ["g4_sample"],
    }
    assert list(matrices["MBL_G3"].columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "g3_sample",
    ]


def test_medulloblastoma_subgroup_assignment_rejects_ties_and_missing_markers():
    tied = _synthetic_mbl_matrix()
    tied.loc[tied["Symbol"].isin(["WIF1", "GLI2"]), "wnt_sample"] = 10.0
    with pytest.raises(ValueError, match=r"maximum is tied.*wnt_sample"):
        expression_builders.medulloblastoma_subgroup_sample_ids(tied)

    missing = _synthetic_mbl_matrix().query("Symbol != 'WIF1'")
    with pytest.raises(ValueError, match="one row for each subgroup marker"):
        expression_builders.medulloblastoma_subgroup_sample_ids(missing)

    non_finite = _synthetic_mbl_matrix()
    non_finite.loc[non_finite["Symbol"].eq("MYC"), "g3_sample"] = np.inf
    with pytest.raises(ValueError, match=r"non-finite.*g3_sample"):
        expression_builders.medulloblastoma_subgroup_sample_ids(non_finite)


def test_sclc_subtype_assignment_keeps_parent_and_routes_one_marker_winner():
    matrix = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            "Symbol": ["ASCL1", "NEUROD1", "POU2F3", "YAP1"],
            "S1": [9.0, 2.0, 1.0, 0.0],
            "S2": [0.0, 8.0, 2.0, 1.0],
            "S3": [1.0, 0.0, 7.0, 2.0],
            "S4": [1.0, 2.0, 0.0, 6.0],
        }
    )

    assert expression_builders.sclc_subtype_sample_ids(matrix) == {
        "SCLC": ["S1", "S2", "S3", "S4"],
        "SCLC_ASCL1": ["S1"],
        "SCLC_NEUROD1": ["S2"],
        "SCLC_POU2F3": ["S3"],
        "SCLC_YAP1": ["S4"],
    }


def test_sclc_subtype_assignment_rejects_ties():
    matrix = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            "Symbol": ["ASCL1", "NEUROD1", "POU2F3", "YAP1"],
            "S1": [9.0, 9.0, 1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="maximum is tied"):
        expression_builders.sclc_subtype_sample_ids(matrix)


def test_sclc_subtype_assignment_takes_maximum_duplicate_source_row():
    matrix = pd.DataFrame(
        {
            "Hugo_Symbol": ["ASCL1", "NEUROD1", "POU2F3", "POU2F3", "YAP1"],
            "S1": [1.0, 10.0, 6.0, 6.0, 0.0],
        }
    )
    matrix.attrs["row_id_col"] = "Hugo_Symbol"
    matrix.attrs["symbol_col"] = None

    routed = expression_builders.sclc_subtype_sample_ids(matrix)

    assert routed["SCLC_NEUROD1"] == ["S1"]
    assert routed["SCLC_POU2F3"] == []


def test_gse75885_titles_route_only_owned_histologies(tmp_path):
    series_matrix = tmp_path / "series.txt.gz"
    with gzip.open(series_matrix, "wt") as handle:
        handle.write(
            '!Sample_title\t"S1 - Liposarcoma - dedifferentiated"'
            '\t"S2 - Low grade fibromyxoid sarcoma"'
            '\t"S3 - Liposarcoma - pleomorphic"'
            '\t"S4 - Leiomyosarcoma"\n'
        )

    titles = expression_source_adapters.geo_sample_titles(series_matrix)

    assert expression_source_adapters.gse75885_routed_samples(
        ["S1", "S2", "S3", "S4"],
        titles,
    ) == {
        "SARC_DDLPS": ["S1"],
        "SARC_PLEOLPS": ["S3"],
        "SARC_LGFMS": ["S2"],
    }


def test_gse75885_routing_requires_metadata_for_every_expression_sample():
    with pytest.raises(ValueError, match="has no GEO title"):
        expression_source_adapters.gse75885_routed_samples(
            ["S1", "S2"],
            {"S1": "Liposarcoma - pleomorphic"},
        )


def test_drmetrics_histology_routing_excludes_sclc():
    attributes = pd.DataFrame(
        {
            "Sample_ID": ["T1", "T2", "T3", "T4", "T5", "SCLC1"],
            "Histopathology_simplified": [
                "Typical",
                "Atypical",
                "Carcinoid",
                "Supra_carcinoid",
                "LCNEC",
                "SCLC",
            ],
        }
    )

    assert expression_source_adapters.drmetrics_routed_samples(
        attributes["Sample_ID"],
        attributes,
    ) == {
        "NET_LUNG": ["T1", "T2", "T3", "T4"],
        "NEC_LUNG_LARGECELL": ["T5"],
    }


def test_drmetrics_histology_routing_rejects_missing_or_duplicate_metadata():
    duplicate = pd.DataFrame(
        {
            "Sample_ID": ["T1", "T1"],
            "Histopathology_simplified": ["Typical", "Atypical"],
        }
    )
    with pytest.raises(ValueError, match="duplicate Sample_ID"):
        expression_source_adapters.drmetrics_routed_samples(["T1"], duplicate)

    attributes = duplicate.drop_duplicates("Sample_ID")
    with pytest.raises(ValueError, match="has no attributes row"):
        expression_source_adapters.drmetrics_routed_samples(["T2"], attributes)


def test_geo_microarray_parsers_keep_characteristics_and_multi_gene_probes(tmp_path):
    series = tmp_path / "series.txt.gz"
    with gzip.open(series, "wt") as handle:
        handle.write('!Sample_title\t"T1"\t"T2"\n')
        handle.write('!Sample_characteristics_ch1\t"subtype: low-grade"\t"subtype: high-grade"\n')
        handle.write("!series_matrix_table_begin\n")
        handle.write('"ID_REF"\t"T1"\t"T2"\n')
        handle.write('"P1"\t"2"\t"3"\n')
        handle.write('"P2"\t"4"\t"1"\n')
        handle.write("!series_matrix_table_end\n")
    platform = tmp_path / "platform.txt"
    platform.write_text(
        "!platform_table_begin\n"
        "ID\tGene Symbol\tENTREZ_GENE_ID\n"
        "P1\tGENE1\t1\n"
        "P2\tGENE2 /// GENE3\t2 /// 3\n"
        "!platform_table_end\n"
    )

    intensities, metadata = expression_source_adapters.parse_geo_series_matrix(series)
    annotations = expression_source_adapters.parse_geo_platform_annotations(platform)

    assert metadata["T1"]["char_subtype"] == "low-grade"
    assert metadata["T2"]["char_subtype"] == "high-grade"
    assert annotations.loc[annotations["probe_id"].eq("P2"), "gene_symbol"].tolist() == [
        "GENE2",
        "GENE3",
    ]
    proxy = expression_source_adapters.microarray_tpm_proxy(
        intensities,
        annotations,
        log2_transformed=True,
    )
    assert np.allclose(proxy.sum(axis=0), 1_000_000.0)
    assert proxy.loc["GENE2", "T1"] > proxy.loc["GENE1", "T1"]


def test_geo_microarray_routing_is_explicit_and_mutually_exclusive():
    metadata = {
        "LG": {"char_subtype": "low-grade endometrial stromal sarcoma"},
        "HG": {"char_subtype": "high-grade endometrial stromal sarcoma"},
        "UUS": {"char_subtype": "undifferentiated uterine sarcoma"},
    }
    patterns = {
        "SARC_ESS_LG": r"(?i)low.?grade",
        "SARC_ESS_HG": r"(?i)high.?grade",
    }

    assert expression_source_adapters.microarray_routed_samples(metadata, patterns) == {
        "SARC_ESS_LG": ["LG"],
        "SARC_ESS_HG": ["HG"],
    }

    with pytest.raises(ValueError, match="matches multiple codes"):
        expression_source_adapters.microarray_routed_samples(
            {"S1": {"subtype": "mixed"}},
            {"A": "mixed", "B": "mixed"},
        )


@pytest.mark.parametrize(
    ("comment", "expected"),
    [
        ("Cell of origin: B-precursor; Risk: standard", ("B-precursor", "B_ALL")),
        ("Cell of origin: T cell", ("T cell", "T_ALL")),
        ("Cell of origin: Indeterminate", ("Indeterminate", "")),
        ("Risk: standard", ("", "")),
    ],
)
def test_target_all_lineage_labels_are_explicit(comment, expected):
    assert expression_source_adapters.parse_target_all_lineage_label(comment) == expected


def test_target_all_registry_uses_complete_gdc_project_ids():
    source = expression_builders.gdc_source_from_registry("target-all")

    assert source.project_ids == ("TARGET-ALL-P1", "TARGET-ALL-P2", "TARGET-ALL-P3")


def test_target_all_lineage_assignments_preserve_evidence(monkeypatch, tmp_path):
    matrix = pd.DataFrame(
        {
            "Case USI": ["TARGET-10-A", "TARGET-10-B", "TARGET-10-X"],
            "Comments": [
                "Cell of origin: B-precursor; other",
                "Cell of origin: T cell",
                "Cell of origin: Indeterminate",
            ],
        }
    )
    monkeypatch.setattr(expression_source_adapters.pd, "read_excel", lambda *_a, **_k: matrix)

    assignments = expression_source_adapters.target_all_lineage_assignments(
        [("TARGET phase sample matrix", tmp_path / "lineage.xlsx")]
    )

    assert assignments["TARGET-10-A"].cancer_code == "B_ALL"
    assert assignments["TARGET-10-B"].cancer_code == "T_ALL"
    assert "Cell of origin: B-precursor" in assignments["TARGET-10-A"].evidence_source
    assert "TARGET-10-X" not in assignments


def test_nbl_mycn_routing_keeps_parent_and_one_child():
    routed = expression_source_adapters.nbl_mycn_routed_samples(
        ["TARGET-30-AMPCASE-01A", "TARGET-30-NONCASE-01A", "TARGET-30-UNKNOWN-02A"],
        {
            "TARGET-30-AMPCASE": "amp",
            "TARGET-30-NONCASE": "nonamp",
            "TARGET-30-UNKNOWN": "unknown",
        },
    )

    assert routed == {
        "NBL": [
            "TARGET-30-AMPCASE-01A",
            "TARGET-30-NONCASE-01A",
            "TARGET-30-UNKNOWN-02A",
        ],
        "NBL_MYCNamp": ["TARGET-30-AMPCASE-01A"],
        "NBL_MYCNnonamp": ["TARGET-30-NONCASE-01A", "TARGET-30-UNKNOWN-02A"],
    }
    with pytest.raises(ValueError, match="lacks a supported MYCN call"):
        expression_source_adapters.nbl_mycn_routed_samples(["TARGET-30-MISSING-01A"], {})


def test_ctcl_pseudobulk_selects_each_cases_dominant_blood_clone(tmp_path):
    raw_tar = tmp_path / "GSE171811_RAW.tar"
    with tarfile.open(raw_tar, "w") as tar:
        for index, case_id in enumerate(["HC1", "SS1", "SS2", "SS3", "SS4", "SS5", "SS6", "MFIV1"]):
            gsm = f"GSM{index + 1}"
            prefix = f"{gsm}_{case_id}_Blood"
            tcr = f"clone\tcell1\tcell2\n{case_id}_dominant\t2\t0\n{case_id}_other\t0\t1\n"
            gex = "gene\tcell1\tcell2\nGENE1\t10\t0\nGENE2\t2\t8\n"
            for suffix, content in (("TCRb", tcr), ("GEX", gex)):
                compressed = gzip.compress(content.encode())
                info = tarfile.TarInfo(f"{prefix}_{suffix}.tsv.gz")
                info.size = len(compressed)
                tar.addfile(info, io.BytesIO(compressed))

    counts, manifest = expression_source_adapters.ctcl_case_pseudobulk(raw_tar)

    assert list(counts.columns) == [
        "source_symbol",
        "MFIV1",
        "SS1",
        "SS2",
        "SS3",
        "SS4",
        "SS5",
        "SS6",
    ]
    assert set(counts["source_symbol"]) == {"GENE1", "GENE2"}
    assert set(counts.set_index("source_symbol").loc["GENE1"]) == {10.0}
    assert manifest["included"].value_counts().to_dict() == {True: 7, False: 1}
    healthy = manifest.loc[~manifest["included"]].iloc[0]
    assert healthy["case_id"] == "HC1"
    assert healthy["exclusion_reason"] == "healthy_control"


def test_ctcl_pseudobulk_requires_identical_gex_and_tcr_cell_columns(tmp_path):
    raw_tar = tmp_path / "GSE171811_RAW.tar"
    with tarfile.open(raw_tar, "w") as tar:
        contents = {
            "GSM1_SS1_Blood_TCRb.tsv.gz": "clone\tcell1\tcell2\ndominant\t1\t0\n",
            "GSM1_SS1_Blood_GEX.tsv.gz": "gene\tcell2\tcell1\nGENE1\t10\t0\n",
        }
        for name, content in contents.items():
            compressed = gzip.compress(content.encode())
            info = tarfile.TarInfo(name)
            info.size = len(compressed)
            tar.addfile(info, io.BytesIO(compressed))

    with pytest.raises(ValueError, match="different cell columns"):
        expression_source_adapters.ctcl_case_pseudobulk(raw_tar)


def test_derive_mbl_subgroup_source_matrices_writes_cache_and_release_assets(tmp_path):
    script = _load_script("derive_mbl_subgroup_source_matrices")
    parent = tmp_path / "MBL.parquet"
    cache_dir = tmp_path / "cache"
    release_dir = tmp_path / "release"
    _synthetic_mbl_matrix().to_parquet(parent, index=False)

    paths = script.derive(parent, output_dir=cache_dir, release_dir=release_dir)

    assert set(paths) == {"MBL_WNT", "MBL_SHH", "MBL_G3", "MBL_G4"}
    for code, path in paths.items():
        assert path == cache_dir / f"{code}.parquet"
        assert path.exists()
        assert (release_dir / f"{code}_per_sample_tpm.parquet").exists()


def test_merge_expression_artifact_rebuild_replaces_only_focused_codes(tmp_path):
    script = _load_script("merge_expression_artifact_rebuild")
    bundle = tmp_path / "bundle"
    rebuild = tmp_path / "rebuild"
    shard_dirs = script._FOCUSED_SHARD_DIRS
    for base in (bundle, rebuild):
        for relative in shard_dirs:
            (base / relative).mkdir(parents=True, exist_ok=True)
        (base / script._REFERENCE_SUMMARY_DIR).mkdir(parents=True, exist_ok=True)

    for code, value in (("KEEP", 1.0), ("REPLACE", 2.0)):
        for relative in shard_dirs:
            pd.DataFrame({"value": [value]}).to_parquet(
                bundle / relative / f"{code}.parquet", index=False
            )
    for relative in shard_dirs:
        pd.DataFrame({"value": [3.0]}).to_parquet(
            rebuild / relative / "REPLACE.parquet", index=False
        )

    pd.DataFrame(
        {"cancer_code": ["KEEP", "REPLACE"], "n_samples": [1, 2], "value": [1.0, 2.0]}
    ).to_csv(bundle / script._REFERENCE_SUMMARY_DIR / "SHARED.csv.gz", index=False)
    pd.DataFrame({"cancer_code": ["REPLACE"], "n_samples": [4], "value": [3.0]}).to_csv(
        rebuild / script._REFERENCE_SUMMARY_DIR / "SHARED.csv.gz", index=False
    )

    pd.DataFrame(
        {
            "representative_id": ["KEEP__rep1", "REPLACE__rep1"],
            "source_cohort": ["KEEP_OLD", "REPLACE_OLD"],
        }
    ).to_csv(bundle / script._REPRESENTATIVE_PROVENANCE, index=False)
    pd.DataFrame(
        {
            "representative_id": ["REPLACE__rep1"],
            "source_cohort": ["REPLACE_NEW"],
        }
    ).to_csv(rebuild / script._REPRESENTATIVE_PROVENANCE, index=False)

    old_metadata = pd.DataFrame(
        {
            "cancer_code": ["KEEP", "REPLACE"],
            "source_cohort": ["KEEP_OLD", "REPLACE_OLD"],
            "n_source_samples": [2, 3],
            "n_cohort_samples": [1, 2],
            "n_negative_values_clipped": [0, 1],
            "sample_qc_fallback_reason": ["", "old"],
        }
    )
    new_metadata = pd.DataFrame(
        {
            "cancer_code": ["REPLACE"],
            "source_cohort": ["REPLACE_NEW"],
            "build_source_cohort": ["REPLACE_BUILD"],
            "n_source_samples": [5],
            "n_cohort_samples": [4],
            "n_negative_values_clipped": [2],
            "sample_qc_fallback_reason": [""],
        }
    )
    old_metadata.to_csv(bundle / script.EXPRESSION_ARTIFACT_BUILD_METADATA_PATH, index=False)
    new_metadata.to_csv(rebuild / script.EXPRESSION_ARTIFACT_BUILD_METADATA_PATH, index=False)
    pd.DataFrame({"cancer_code": ["KEEP", "REPLACE"], "sample_id": ["a", "b"]}).to_csv(
        bundle / script.SOURCE_MATRIX_SAMPLE_QC_MANIFEST_PATH, index=False
    )
    pd.DataFrame({"cancer_code": ["REPLACE"], "sample_id": ["c"]}).to_csv(
        rebuild / script.SOURCE_MATRIX_SAMPLE_QC_MANIFEST_PATH, index=False
    )
    (bundle / script.EXPRESSION_ARTIFACT_BUILD_METADATA_JSON_PATH).write_text("{}\n")

    assert script.merge(bundle, rebuild) == {"REPLACE"}
    assert (
        pd.read_parquet(bundle / "cancer-reference-expression-percentiles" / "KEEP.parquet").loc[
            0, "value"
        ]
        == 1.0
    )
    assert (
        pd.read_parquet(bundle / "cancer-reference-expression-percentiles" / "REPLACE.parquet").loc[
            0, "value"
        ]
        == 3.0
    )
    merged_metadata = pd.read_csv(bundle / script.EXPRESSION_ARTIFACT_BUILD_METADATA_PATH)
    assert set(merged_metadata["cancer_code"]) == {"KEEP", "REPLACE"}
    indexed_metadata = merged_metadata.set_index("cancer_code")
    assert indexed_metadata.loc["REPLACE", "n_source_samples"] == 5
    assert indexed_metadata.loc["KEEP", "build_source_cohort"] == "KEEP_OLD"
    assert indexed_metadata.loc["REPLACE", "build_source_cohort"] == "REPLACE_BUILD"
    merged_reference = pd.read_csv(bundle / script._REFERENCE_SUMMARY_DIR / "SHARED.csv.gz")
    merged_reference = merged_reference.set_index("cancer_code")
    assert merged_reference.loc["KEEP", "n_samples"] == 1
    assert merged_reference.loc["REPLACE", "n_samples"] == 4
    assert merged_reference.loc["REPLACE", "value"] == 3.0
    summary = json.loads((bundle / script.EXPRESSION_ARTIFACT_BUILD_METADATA_JSON_PATH).read_text())
    assert summary["n_cohorts"] == 2
    assert summary["n_source_samples"] == 7
    assert summary["n_cohort_samples"] == 5
    assert summary["n_negative_values_clipped"] == 2
    assert summary["sample_qc_fallbacks"] == 0
    assert script._REFERENCE_SUMMARY_DIR in summary["derived_artifacts"]


def test_merge_expression_artifact_rebuild_requires_every_shard_family(tmp_path):
    script = _load_script("merge_expression_artifact_rebuild")
    bundle = tmp_path / "bundle"
    rebuild = tmp_path / "rebuild"
    missing_dir = script._FOCUSED_SHARD_DIRS[-1]
    for relative in script._FOCUSED_SHARD_DIRS[:-1]:
        directory = rebuild / relative
        directory.mkdir(parents=True)
        pd.DataFrame({"value": [1.0]}).to_parquet(directory / "X.parquet", index=False)

    with pytest.raises(FileNotFoundError, match=f"artifact directory: {missing_dir}"):
        script._copy_rebuilt_shards(bundle, rebuild, cancer_codes={"X"})
    assert not bundle.exists()


def test_merge_expression_artifact_rebuild_replaces_preferred_code_specific_summary(
    tmp_path,
):
    script = _load_script("merge_expression_artifact_rebuild")
    bundle = tmp_path / "bundle"
    rebuild = tmp_path / "rebuild"
    bundle_summary = bundle / script._REFERENCE_SUMMARY_DIR
    rebuild_summary = rebuild / script._REFERENCE_SUMMARY_DIR
    bundle_summary.mkdir(parents=True)
    rebuild_summary.mkdir(parents=True)

    pd.DataFrame({"cancer_code": ["REPLACE"], "source_cohort": ["SHARED"], "value": [1.0]}).to_csv(
        bundle_summary / "SHARED__REPLACE.csv.gz", index=False
    )
    pd.DataFrame(
        {
            "cancer_code": ["KEEP", "REPLACE"],
            "source_cohort": ["SHARED", "SHARED"],
            "value": [2.0, 1.0],
        }
    ).to_csv(bundle_summary / "SHARED.csv.gz", index=False)
    pd.DataFrame({"cancer_code": ["REPLACE"], "source_cohort": ["SHARED"], "value": [3.0]}).to_csv(
        rebuild_summary / "SHARED.csv.gz", index=False
    )

    script._merge_reference_summaries(bundle, rebuild, cancer_codes={"REPLACE"})

    specific = pd.read_csv(bundle_summary / "SHARED__REPLACE.csv.gz")
    consolidated = pd.read_csv(bundle_summary / "SHARED.csv.gz")
    assert specific[["cancer_code", "value"]].to_dict("records") == [
        {"cancer_code": "REPLACE", "value": 3.0}
    ]
    assert consolidated[["cancer_code", "value"]].to_dict("records") == [
        {"cancer_code": "KEEP", "value": 2.0}
    ]


def test_stage_source_matrices_can_reuse_a_prior_version_cache(tmp_path, monkeypatch):
    script = _load_script("stage_source_matrices")
    builder_cache = tmp_path / "builder-cache"
    existing_cache = tmp_path / "source-v-old"
    active_cache = tmp_path / "source-v-new"
    release_dir = tmp_path / "release"
    historical_dir = builder_cache / "historical-name" / "derived"
    historical_dir.mkdir(parents=True)
    existing_cache.mkdir()
    matrix = pd.DataFrame({"Ensembl_Gene_ID": ["E1"], "Symbol": ["G1"], "sample": [1.0]})
    matrix.to_parquet(existing_cache / "X.parquet", index=False)
    matrix.to_parquet(historical_dir / "Y_per_sample_tpm.parquet", index=False)
    monkeypatch.setattr(
        script.sm,
        "registry",
        lambda: pd.DataFrame(
            {
                "cancer_code": ["X", "Y"],
                "source_cohort": ["RENAMED_SOURCE", "OTHER_RENAMED_SOURCE"],
            }
        ),
    )
    monkeypatch.setattr(script.sm, "cache_dir", lambda: active_cache)

    script.stage(
        builder_cache,
        release_dir=release_dir,
        codes=None,
        limit=None,
        existing_cache=existing_cache,
    )

    assert (active_cache / "X.parquet").exists()
    assert (release_dir / "X_per_sample_tpm.parquet").exists()
    assert (active_cache / "Y.parquet").exists()
    assert (release_dir / "Y_per_sample_tpm.parquet").exists()


def test_stage_source_matrices_does_not_misroute_partial_shared_source(tmp_path, monkeypatch):
    script = _load_script("stage_source_matrices")
    builder_cache = tmp_path / "builder-cache"
    source_dir = builder_cache / "shared-source" / "derived"
    existing_cache = tmp_path / "source-v-old"
    active_cache = tmp_path / "source-v-new"
    source_dir.mkdir(parents=True)
    existing_cache.mkdir()
    x = pd.DataFrame({"Ensembl_Gene_ID": ["E1"], "Symbol": ["G1"], "x": [1.0]})
    y = pd.DataFrame({"Ensembl_Gene_ID": ["E1"], "Symbol": ["G1"], "y": [2.0]})
    x.to_parquet(source_dir / "X_per_sample_tpm.parquet", index=False)
    y.to_parquet(existing_cache / "Y.parquet", index=False)
    monkeypatch.setattr(
        script.sm,
        "registry",
        lambda: pd.DataFrame(
            {
                "cancer_code": ["X", "Y"],
                "source_cohort": ["SHARED_SOURCE", "SHARED_SOURCE"],
            }
        ),
    )
    monkeypatch.setattr(script.sm, "cache_dir", lambda: active_cache)

    script.stage(
        builder_cache,
        release_dir=None,
        codes=None,
        limit=None,
        existing_cache=existing_cache,
    )

    assert list(pd.read_parquet(active_cache / "X.parquet").columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "x",
    ]
    assert list(pd.read_parquet(active_cache / "Y.parquet").columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "y",
    ]


def test_stage_source_matrices_rejects_missing_or_wrong_width_assets(tmp_path, monkeypatch):
    script = _load_script("stage_source_matrices")
    builder_cache = tmp_path / "builder-cache"
    source_dir = builder_cache / "source-x" / "derived"
    active_cache = tmp_path / "source-v-new"
    source_dir.mkdir(parents=True)
    pd.DataFrame({"Ensembl_Gene_ID": ["E1"], "Symbol": ["G1"], "only_sample": [1.0]}).to_parquet(
        source_dir / "X_per_sample_tpm.parquet", index=False
    )
    monkeypatch.setattr(script.sm, "cache_dir", lambda: active_cache)

    monkeypatch.setattr(
        script.sm,
        "registry",
        lambda: pd.DataFrame(
            {
                "cancer_code": ["X"],
                "source_cohort": ["SOURCE_X"],
                "n_samples": [2],
            }
        ),
    )
    with pytest.raises(ValueError, match="selected matrix has 1 samples; registry expects 2"):
        script.stage(builder_cache, release_dir=None, codes=None, limit=None)

    monkeypatch.setattr(
        script.sm,
        "registry",
        lambda: pd.DataFrame(
            {
                "cancer_code": ["X", "Y"],
                "source_cohort": ["SOURCE_X", "SOURCE_Y"],
            }
        ),
    )
    with pytest.raises(FileNotFoundError, match="missing for 1 cohort"):
        script.stage(builder_cache, release_dir=None, codes=None, limit=None)
    assert not list(active_cache.glob("*.parquet"))


def test_treehouse_source_from_registry_loads_tcga_sample_routes():
    source = expression_builders.treehouse_source_from_registry("treehouse-polya-25-01-tcga-subset")

    assert source.source_cohort == "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES"
    assert source.source_project == "Treehouse (TCGA samples)"
    assert source.pipeline_stem == "treehouse_polya_25_01_tcga_samples"
    assert isinstance(source.cancer_code, list)
    assert len(source.cancer_code) == 30
    assert len(source.cohorts) == 30
    by_code = {cohort.cancer_code: cohort for cohort in source.cohorts}
    assert by_code["BRCA"].disease_label == "breast invasive carcinoma"
    assert by_code["BRCA"].selection == "tcga"
    assert by_code["BRCA"].effective_cache_stem == "tcga_brca"
    assert by_code["UCEC"].disease_label == "uterine corpus endometrioid carcinoma"
    assert "GBM" not in by_code
    assert "LGG" not in by_code
    assert "SARC" not in by_code

    tcga_direct = expression_builders.treehouse_cohorts_for_group(
        "tcga_direct",
        source_id="treehouse-polya-25-01-tcga-subset",
    )
    assert [cohort.cancer_code for cohort in tcga_direct] == source.cancer_code
    assert {cohort.selection for cohort in tcga_direct} == {"tcga"}


def test_treehouse_source_from_registry_loads_brca_pam50_routes():
    source = expression_builders.treehouse_source_from_registry(
        "treehouse-polya-25-01-tcga-brca-pam50"
    )

    assert source.source_cohort == "TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50"
    assert source.pipeline_stem == "treehouse_polya_25_01_tcga_brca_pam50"
    assert source.cancer_code == [
        "BRCA_Basal",
        "BRCA_HER2",
        "BRCA_LumA",
        "BRCA_LumB",
        "BRCA_Normal",
    ]
    by_code = {cohort.cancer_code: cohort for cohort in source.cohorts}
    assert by_code["BRCA_Basal"].disease_label == "breast invasive carcinoma"
    assert (
        by_code["BRCA_Basal"].selection
        == "cbio_clinical:brca_tcga_pan_can_atlas_2018:SUBTYPE:BRCA_Basal"
    )
    assert (
        by_code["BRCA_HER2"].selection
        == "cbio_clinical:brca_tcga_pan_can_atlas_2018:SUBTYPE:BRCA_Her2"
    )
    assert by_code["BRCA_HER2"].effective_cache_stem == "tcga_brca_her2"

    pam50 = expression_builders.treehouse_cohorts_for_group(
        "tcga_brca_pam50",
        source_id="treehouse-polya-25-01-tcga-brca-pam50",
    )
    assert [cohort.cancer_code for cohort in pam50] == source.cancer_code


def test_treehouse_source_from_registry_loads_hnsc_hpv_routes():
    source = expression_builders.treehouse_source_from_registry(
        "treehouse-polya-25-01-tcga-hnsc-hpv"
    )

    assert source.source_cohort == "TREEHOUSE_POLYA_25_01_TCGA_HNSC_HPV"
    assert source.pipeline_stem == "treehouse_polya_25_01_tcga_hnsc_hpv"
    assert source.cancer_code == ["HNSC_HPVneg", "HNSC_HPVpos"]
    by_code = {cohort.cancer_code: cohort for cohort in source.cohorts}
    assert by_code["HNSC_HPVneg"].disease_label == "head & neck squamous cell carcinoma"
    assert (
        by_code["HNSC_HPVneg"].selection
        == "cbio_clinical:hnsc_tcga_pan_can_atlas_2018:SUBTYPE:HNSC_HPV-"
    )
    assert (
        by_code["HNSC_HPVpos"].selection
        == "cbio_clinical:hnsc_tcga_pan_can_atlas_2018:SUBTYPE:HNSC_HPV+"
    )
    assert by_code["HNSC_HPVpos"].effective_cache_stem == "tcga_hnsc_hpvpos"

    hpv = expression_builders.treehouse_cohorts_for_group(
        "tcga_hnsc_hpv",
        source_id="treehouse-polya-25-01-tcga-hnsc-hpv",
    )
    assert [cohort.cancer_code for cohort in hpv] == source.cancer_code


def test_treehouse_source_from_registry_loads_luad_mutation_routes():
    source = expression_builders.treehouse_source_from_registry(
        "treehouse-polya-25-01-tcga-luad-mut"
    )

    assert source.source_cohort == "TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT"
    assert source.pipeline_stem == "treehouse_polya_25_01_tcga_luad_mut"
    assert source.cancer_code == ["LUAD_EGFR", "LUAD_KRAS", "LUAD_STK11"]
    by_code = {cohort.cancer_code: cohort for cohort in source.cohorts}
    assert by_code["LUAD_EGFR"].disease_label == "lung adenocarcinoma"
    assert by_code["LUAD_EGFR"].selection == "cbio_mutation:luad_tcga_pan_can_atlas_2018:EGFR"
    assert (
        by_code["LUAD_STK11"].selection == "cbio_mutation:luad_tcga_pan_can_atlas_2018:STK11,KEAP1"
    )
    assert by_code["LUAD_STK11"].effective_cache_stem == "tcga_luad_stk11"

    luad_mut = expression_builders.treehouse_cohorts_for_group(
        "tcga_luad_mut",
        source_id="treehouse-polya-25-01-tcga-luad-mut",
    )
    assert [cohort.cancer_code for cohort in luad_mut] == source.cancer_code

    registry = cohort_registry()
    assert registry["TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT"]["n_samples"] == 362


def test_treehouse_source_from_registry_loads_sarc_histology_routes():
    source = expression_builders.treehouse_source_from_registry(
        "treehouse-polya-25-01-tcga-sarc-histology"
    )

    assert source.source_cohort == "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY"
    assert source.pipeline_stem == "treehouse_polya_25_01_tcga_sarc_histology"
    assert source.cancer_code == ["SARC_WDLPS", "SARC_DDLPS", "SARC_PLEOLPS"]
    by_code = {cohort.cancer_code: cohort for cohort in source.cohorts}
    assert by_code["SARC_WDLPS"].disease_label == "liposarcoma"
    assert (
        by_code["SARC_WDLPS"].selection
        == "gdc_primary_diagnosis:TCGA-SARC:Liposarcoma, well differentiated"
    )
    assert (
        by_code["SARC_DDLPS"].selection
        == "gdc_primary_diagnosis:TCGA-SARC:Dedifferentiated liposarcoma"
    )
    assert (
        by_code["SARC_PLEOLPS"].selection
        == "gdc_primary_diagnosis:TCGA-SARC:Pleomorphic liposarcoma"
    )

    sarc = expression_builders.treehouse_cohorts_for_group(
        "tcga_sarc_histology",
        source_id="treehouse-polya-25-01-tcga-sarc-histology",
    )
    assert [cohort.cancer_code for cohort in sarc] == source.cancer_code

    registry = cohort_registry()
    assert registry["TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY"]["n_samples"] == 55
    assert registry["TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY"]["n_codes"] == 3


def test_treehouse_source_from_registry_loads_ucec_subtype_routes():
    source = expression_builders.treehouse_source_from_registry(
        "treehouse-polya-25-01-tcga-ucec-subtype"
    )

    assert source.source_cohort == "TREEHOUSE_POLYA_25_01_TCGA_UCEC_SUBTYPE"
    assert source.pipeline_stem == "treehouse_polya_25_01_tcga_ucec_subtype"
    assert source.cancer_code == ["UCEC_POLE", "UCEC_MSI", "UCEC_CNL", "UCEC_CNH"]
    by_code = {cohort.cancer_code: cohort for cohort in source.cohorts}
    assert by_code["UCEC_POLE"].disease_label == "endometrial carcinoma"
    assert (
        by_code["UCEC_POLE"].selection
        == "cbio_clinical:ucec_tcga_pan_can_atlas_2018:SUBTYPE:UCEC_POLE"
    )
    assert (
        by_code["UCEC_CNL"].selection
        == "cbio_clinical:ucec_tcga_pan_can_atlas_2018:SUBTYPE:UCEC_CN_LOW"
    )
    assert (
        by_code["UCEC_CNH"].selection
        == "cbio_clinical:ucec_tcga_pan_can_atlas_2018:SUBTYPE:UCEC_CN_HIGH"
    )
    assert by_code["UCEC_CNH"].effective_cache_stem == "tcga_ucec_cnh"

    subtypes = expression_builders.treehouse_cohorts_for_group(
        "tcga_ucec_subtype",
        source_id="treehouse-polya-25-01-tcga-ucec-subtype",
    )
    assert [cohort.cancer_code for cohort in subtypes] == source.cancer_code

    registry = cohort_registry()
    assert registry["TREEHOUSE_POLYA_25_01_TCGA_UCEC_SUBTYPE"]["n_samples"] == 172


def test_treehouse_source_from_registry_loads_stad_subtype_routes():
    source = expression_builders.treehouse_source_from_registry(
        "treehouse-polya-25-01-tcga-stad-subtype"
    )

    assert source.source_cohort == "TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE"
    assert source.pipeline_stem == "treehouse_polya_25_01_tcga_stad_subtype"
    assert source.cancer_code == ["STAD_EBV", "STAD_MSI", "STAD_GS", "STAD_CIN"]
    by_code = {cohort.cancer_code: cohort for cohort in source.cohorts}
    assert by_code["STAD_EBV"].disease_label == "stomach adenocarcinoma"
    assert (
        by_code["STAD_EBV"].selection
        == "cbio_clinical:stad_tcga_pan_can_atlas_2018:SUBTYPE:STAD_EBV"
    )
    assert (
        by_code["STAD_CIN"].selection
        == "cbio_clinical:stad_tcga_pan_can_atlas_2018:SUBTYPE:STAD_CIN"
    )
    assert by_code["STAD_CIN"].effective_cache_stem == "tcga_stad_cin"

    subtypes = expression_builders.treehouse_cohorts_for_group(
        "tcga_stad_subtype",
        source_id="treehouse-polya-25-01-tcga-stad-subtype",
    )
    assert [cohort.cancer_code for cohort in subtypes] == source.cancer_code

    registry = cohort_registry()
    assert registry["TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE"]["n_samples"] == 374


def test_treehouse_source_from_registry_loads_coadread_msi_routes():
    source = expression_builders.treehouse_source_from_registry(
        "treehouse-polya-25-01-tcga-coadread-msi"
    )

    assert source.source_cohort == "TREEHOUSE_POLYA_25_01_TCGA_COADREAD_MSI"
    assert source.pipeline_stem == "treehouse_polya_25_01_tcga_coadread_msi"
    assert source.cancer_code == ["COAD_MSI", "COAD_MSS", "READ_MSI", "READ_MSS"]
    by_code = {cohort.cancer_code: cohort for cohort in source.cohorts}
    assert by_code["COAD_MSI"].disease_label == "colon adenocarcinoma"
    assert by_code["READ_MSI"].disease_label == "rectum adenocarcinoma"
    assert (
        by_code["COAD_MSI"].selection
        == "cbio_sample_clinical:coadread_tcga_pan_can_atlas_2018:MSI_SENSOR_SCORE:>=10"
    )
    assert (
        by_code["COAD_MSS"].selection
        == "cbio_sample_clinical:coadread_tcga_pan_can_atlas_2018:MSI_SENSOR_SCORE:<10"
    )
    assert by_code["READ_MSS"].effective_cache_stem == "tcga_read_mss"

    subtypes = expression_builders.treehouse_cohorts_for_group(
        "tcga_coadread_msi",
        source_id="treehouse-polya-25-01-tcga-coadread-msi",
    )
    assert [cohort.cancer_code for cohort in subtypes] == source.cancer_code

    registry = cohort_registry()
    assert registry["TREEHOUSE_POLYA_25_01_TCGA_COADREAD_MSI"]["n_samples"] == 361


def test_treehouse_source_from_registry_loads_glioma_gdc_project_routes():
    source = expression_builders.treehouse_source_from_registry("treehouse-polya-25-01-tcga-glioma")

    assert source.source_cohort == "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES"
    assert source.pipeline_stem == "treehouse_polya_25_01_tcga_glioma_split"
    assert source.cancer_code == ["GBM", "LGG"]
    by_code = {cohort.cancer_code: cohort for cohort in source.cohorts}
    assert by_code["GBM"].disease_label == "glioma"
    assert by_code["GBM"].selection == "gdc_project:TCGA-GBM"
    assert by_code["GBM"].effective_cache_stem == "tcga_gbm"
    assert by_code["LGG"].selection == "gdc_project:TCGA-LGG"

    glioma = expression_builders.treehouse_cohorts_for_group(
        "tcga_glioma",
        source_id="treehouse-polya-25-01-tcga-glioma",
    )
    assert [cohort.cancer_code for cohort in glioma] == ["GBM", "LGG"]


def test_treehouse_sample_ids_filter_disease_and_tcga_selection():
    clinical = pd.DataFrame(
        [
            {"th_dataset_id": "TREEHOUSE-1", "disease": "Ewing sarcoma"},
            {"th_dataset_id": "TREEHOUSE-2", "disease": "ewing sarcoma"},
            {"th_dataset_id": "TCGA-AB-1234-01A", "disease": "Ewing sarcoma"},
            {"th_dataset_id": "TCGA-XY-9999-01A", "disease": "osteosarcoma"},
        ]
    )
    cohort = expression_builders.TreehouseCohort("SARC_EWS", "Ewing sarcoma")
    assert expression_builders.treehouse_sample_ids(clinical, cohort) == [
        "TREEHOUSE-1",
        "TREEHOUSE-2",
        "TCGA-AB-1234-01A",
    ]

    tcga = expression_builders.TreehouseCohort(
        "SARC_EWS",
        "Ewing sarcoma",
        selection="tcga",
    )
    assert expression_builders.treehouse_sample_ids(clinical, tcga) == ["TCGA-AB-1234-01A"]

    unsupported = expression_builders.TreehouseCohort(
        "SARC_EWS",
        "Ewing sarcoma",
        selection="legacy_pam50:BRCA_Basal",
    )
    with pytest.raises(ValueError, match="unsupported Treehouse cohort selection"):
        expression_builders.treehouse_sample_ids(clinical, unsupported)


def test_treehouse_sample_ids_filter_gdc_project_selection():
    clinical = pd.DataFrame(
        [
            {"th_dataset_id": "TCGA-GB-0001-01A", "disease": "glioma"},
            {"th_dataset_id": "TCGA-LG-0002-01A", "disease": "glioma"},
            {"th_dataset_id": "TREEHOUSE-3", "disease": "glioma"},
        ]
    )
    cohort = expression_builders.TreehouseCohort(
        "GBM",
        "glioma",
        selection="gdc_project:TCGA-GBM",
    )
    case_sets = {"gdc_project:TCGA-GBM": {"TCGA-GB-0001"}}

    assert expression_builders.treehouse_sample_ids(
        clinical,
        cohort,
        selection_case_sets=case_sets,
    ) == ["TCGA-GB-0001-01A"]

    with pytest.raises(ValueError, match="requires a precomputed GDC case set"):
        expression_builders.treehouse_sample_ids(clinical, cohort)


def test_treehouse_sample_ids_filter_gdc_primary_diagnosis_selection():
    clinical = pd.DataFrame(
        [
            {"th_dataset_id": "TCGA-DX-0001-01A", "disease": "liposarcoma"},
            {"th_dataset_id": "TCGA-DX-0002-01A", "disease": "liposarcoma"},
            {"th_dataset_id": "TREEHOUSE-3", "disease": "liposarcoma"},
        ]
    )
    selector = "gdc_primary_diagnosis:TCGA-SARC:Liposarcoma, well differentiated"
    cohort = expression_builders.TreehouseCohort(
        "SARC_WDLPS",
        "liposarcoma",
        selection=selector,
    )
    case_sets = {selector: {"TCGA-DX-0001"}}

    assert expression_builders.treehouse_sample_ids(
        clinical,
        cohort,
        selection_case_sets=case_sets,
    ) == ["TCGA-DX-0001-01A"]

    with pytest.raises(ValueError, match="requires a precomputed GDC diagnosis case set"):
        expression_builders.treehouse_sample_ids(clinical, cohort)


def test_treehouse_sample_ids_filter_cbio_clinical_selection():
    clinical = pd.DataFrame(
        [
            {"th_dataset_id": "TCGA-BR-0001-01A", "disease": "breast invasive carcinoma"},
            {"th_dataset_id": "TCGA-BR-0002-01A", "disease": "breast invasive carcinoma"},
            {"th_dataset_id": "TREEHOUSE-3", "disease": "breast invasive carcinoma"},
        ]
    )
    selector = "cbio_clinical:brca_tcga_pan_can_atlas_2018:SUBTYPE:BRCA_Basal"
    cohort = expression_builders.TreehouseCohort(
        "BRCA_Basal",
        "breast invasive carcinoma",
        selection=selector,
    )
    case_sets = {selector: {"TCGA-BR-0001"}}

    assert expression_builders.treehouse_sample_ids(
        clinical,
        cohort,
        selection_case_sets=case_sets,
    ) == ["TCGA-BR-0001-01A"]

    with pytest.raises(ValueError, match="requires a precomputed cBioPortal case set"):
        expression_builders.treehouse_sample_ids(clinical, cohort)


def test_treehouse_sample_ids_filter_cbio_sample_clinical_numeric_selection():
    clinical = pd.DataFrame(
        [
            {"th_dataset_id": "TCGA-CO-0001-01A", "disease": "colon adenocarcinoma"},
            {"th_dataset_id": "TCGA-CO-0002-01A", "disease": "colon adenocarcinoma"},
            {"th_dataset_id": "TCGA-RE-0003-01A", "disease": "rectum adenocarcinoma"},
        ]
    )
    selector = "cbio_sample_clinical:coadread_tcga_pan_can_atlas_2018:MSI_SENSOR_SCORE:>=10"
    cohort = expression_builders.TreehouseCohort(
        "COAD_MSI",
        "colon adenocarcinoma",
        selection=selector,
    )
    case_sets = {selector: {"TCGA-CO-0001", "TCGA-RE-0003"}}

    assert expression_builders.treehouse_sample_ids(
        clinical,
        cohort,
        selection_case_sets=case_sets,
    ) == ["TCGA-CO-0001-01A"]

    with pytest.raises(ValueError, match="requires a precomputed cBioPortal case set"):
        expression_builders.treehouse_sample_ids(clinical, cohort)


def test_treehouse_sample_ids_filter_cbio_mutation_selection():
    clinical = pd.DataFrame(
        [
            {"th_dataset_id": "TCGA-LU-0001-01A", "disease": "lung adenocarcinoma"},
            {"th_dataset_id": "TCGA-LU-0002-01A", "disease": "lung adenocarcinoma"},
            {"th_dataset_id": "TREEHOUSE-3", "disease": "lung adenocarcinoma"},
        ]
    )
    selector = "cbio_mutation:luad_tcga_pan_can_atlas_2018:STK11,KEAP1"
    cohort = expression_builders.TreehouseCohort(
        "LUAD_STK11",
        "lung adenocarcinoma",
        selection=selector,
    )
    case_sets = {selector: {"TCGA-LU-0002"}}

    assert expression_builders.treehouse_sample_ids(
        clinical,
        cohort,
        selection_case_sets=case_sets,
    ) == ["TCGA-LU-0002-01A"]

    with pytest.raises(ValueError, match="requires a precomputed cBioPortal case set"):
        expression_builders.treehouse_sample_ids(clinical, cohort)


def test_build_treehouse_source_matrices_writes_canonical_artifacts(tmp_path):
    clinical_path = tmp_path / "clinical.tsv"
    pd.DataFrame(
        [
            {"th_dataset_id": "SAMPLE_A", "disease": "synthetic tumor"},
            {"th_dataset_id": "SAMPLE_B", "disease": "Synthetic Tumor"},
            {"th_dataset_id": "CONTROL", "disease": "other"},
        ]
    ).to_csv(clinical_path, sep="\t", index=False)
    tpm_path = tmp_path / "treehouse.tsv"
    pd.DataFrame(
        {
            "Gene": ["TP53", "EGFR"],
            "SAMPLE_A": np.log2(np.array([2.0, 1.0]) + 1.0),
            "SAMPLE_B": np.log2(np.array([4.0, 0.0]) + 1.0),
            "CONTROL": np.log2(np.array([100.0, 100.0]) + 1.0),
        }
    ).to_csv(tpm_path, sep="\t", index=False)
    source = expression_builders.TreehouseSource(
        source_id="synthetic-treehouse",
        source_cohort="SYNTHETIC_TREEHOUSE",
        cancer_code="CODE_A",
        tpm_file=tpm_path.name,
        clinical_file=clinical_path.name,
        source_project="Treehouse",
        cohorts=(
            expression_builders.TreehouseCohort(
                "CODE_A",
                "synthetic tumor",
                extra_notes="Synthetic cohort note.",
            ),
        ),
        notes="Synthetic source note.",
        pipeline_stem="synthetic_treehouse",
    )

    result = expression_builders.build_treehouse_source_matrices(source, cache_dir=tmp_path)

    assert set(result.matrix_paths) == {"CODE_A"}
    out = pd.read_parquet(result.matrix_paths["CODE_A"]).set_index("Symbol")
    assert list(out.columns) == ["Ensembl_Gene_ID", "SAMPLE_A", "SAMPLE_B"]
    np.testing.assert_allclose(
        out.loc["TP53", ["SAMPLE_A", "SAMPLE_B"]].astype(float).to_numpy(),
        [2.0, 4.0],
    )
    np.testing.assert_allclose(
        out.loc["EGFR", ["SAMPLE_A", "SAMPLE_B"]].astype(float).to_numpy(),
        [1.0, 0.0],
    )
    assert result.sidecar_paths["mapping_audit"].exists()
    assert result.sidecar_paths["parse_diagnostics"].exists()
    assert result.sidecar_paths["summary_rows"].exists()
    assert result.summary_rows["notes"].str.contains("Synthetic cohort note").all()
    summary = result.summary_rows.set_index("Symbol")
    assert summary.loc["TP53", "TPM_median"] == 3.0
    assert set(result.sample_qc["sample_id"]) == {"SAMPLE_A", "SAMPLE_B"}


def test_build_treehouse_source_matrices_splits_gdc_project_cohorts(
    tmp_path,
    monkeypatch,
):
    clinical_path = tmp_path / "clinical.tsv"
    pd.DataFrame(
        [
            {"th_dataset_id": "TCGA-GB-0001-01A", "disease": "glioma"},
            {"th_dataset_id": "TCGA-LG-0002-01A", "disease": "glioma"},
            {"th_dataset_id": "TREEHOUSE-3", "disease": "glioma"},
        ]
    ).to_csv(clinical_path, sep="\t", index=False)
    tpm_path = tmp_path / "treehouse.tsv"
    pd.DataFrame(
        {
            "Gene": ["TP53", "EGFR"],
            "TCGA-GB-0001-01A": np.log2(np.array([2.0, 1.0]) + 1.0),
            "TCGA-LG-0002-01A": np.log2(np.array([4.0, 0.0]) + 1.0),
            "TREEHOUSE-3": np.log2(np.array([100.0, 100.0]) + 1.0),
        }
    ).to_csv(tpm_path, sep="\t", index=False)
    source = expression_builders.TreehouseSource(
        source_id="synthetic-treehouse-glioma",
        source_cohort="SYNTHETIC_TREEHOUSE_TCGA",
        cancer_code=["GBM", "LGG"],
        tpm_file=tpm_path.name,
        clinical_file=clinical_path.name,
        source_project="Treehouse (TCGA samples)",
        cohorts=(
            expression_builders.TreehouseCohort(
                "GBM",
                "glioma",
                selection="gdc_project:TCGA-GBM",
                cache_stem="tcga_gbm",
            ),
            expression_builders.TreehouseCohort(
                "LGG",
                "glioma",
                selection="gdc_project:TCGA-LGG",
                cache_stem="tcga_lgg",
            ),
        ),
    )

    def fake_case_map(project_ids, *, cache_path=None, force_download=False):
        assert set(project_ids) == {"TCGA-GBM", "TCGA-LGG"}
        assert cache_path is not None
        return pd.DataFrame(
            {
                "submitter_id": ["TCGA-GB-0001", "TCGA-LG-0002"],
                "project_id": ["TCGA-GBM", "TCGA-LGG"],
            }
        )

    monkeypatch.setattr(
        expression_builders,
        "treehouse_gdc_case_project_map",
        fake_case_map,
    )
    result = expression_builders.build_treehouse_source_matrices(source, cache_dir=tmp_path)

    assert set(result.matrix_paths) == {"GBM", "LGG"}
    gbm = pd.read_parquet(result.matrix_paths["GBM"]).set_index("Symbol")
    lgg = pd.read_parquet(result.matrix_paths["LGG"]).set_index("Symbol")
    assert list(gbm.columns) == ["Ensembl_Gene_ID", "TCGA-GB-0001-01A"]
    assert list(lgg.columns) == ["Ensembl_Gene_ID", "TCGA-LG-0002-01A"]
    np.testing.assert_allclose(gbm.loc["TP53", "TCGA-GB-0001-01A"], 2.0)
    np.testing.assert_allclose(lgg.loc["TP53", "TCGA-LG-0002-01A"], 4.0)


def test_build_treehouse_source_matrices_splits_gdc_primary_diagnosis_cohorts(
    tmp_path,
    monkeypatch,
):
    clinical_path = tmp_path / "clinical.tsv"
    pd.DataFrame(
        [
            {"th_dataset_id": "TCGA-DX-0001-01A", "disease": "liposarcoma"},
            {"th_dataset_id": "TCGA-DX-0002-01A", "disease": "liposarcoma"},
            {"th_dataset_id": "TCGA-DX-0003-01A", "disease": "liposarcoma"},
            {"th_dataset_id": "TREEHOUSE-4", "disease": "liposarcoma"},
        ]
    ).to_csv(clinical_path, sep="\t", index=False)
    tpm_path = tmp_path / "treehouse.tsv"
    pd.DataFrame(
        {
            "Gene": ["TP53", "MDM2"],
            "TCGA-DX-0001-01A": np.log2(np.array([2.0, 1.0]) + 1.0),
            "TCGA-DX-0002-01A": np.log2(np.array([4.0, 8.0]) + 1.0),
            "TCGA-DX-0003-01A": np.log2(np.array([6.0, 3.0]) + 1.0),
            "TREEHOUSE-4": np.log2(np.array([100.0, 100.0]) + 1.0),
        }
    ).to_csv(tpm_path, sep="\t", index=False)
    source = expression_builders.TreehouseSource(
        source_id="synthetic-treehouse-sarc-histology",
        source_cohort="SYNTHETIC_TREEHOUSE_SARC_HISTOLOGY",
        cancer_code=["SARC_WDLPS", "SARC_DDLPS"],
        tpm_file=tpm_path.name,
        clinical_file=clinical_path.name,
        source_project="Treehouse (TCGA-SARC) x GDC primary diagnosis",
        cohorts=(
            expression_builders.TreehouseCohort(
                "SARC_WDLPS",
                "liposarcoma",
                selection="gdc_primary_diagnosis:TCGA-SARC:Liposarcoma, well differentiated",
                cache_stem="tcga_sarc_wdlps",
            ),
            expression_builders.TreehouseCohort(
                "SARC_DDLPS",
                "liposarcoma",
                selection="gdc_primary_diagnosis:TCGA-SARC:Dedifferentiated liposarcoma",
                cache_stem="tcga_sarc_ddlps",
            ),
        ),
    )

    def fake_diagnosis_map(project_id, *, cache_path=None, force_download=False):
        assert project_id == "TCGA-SARC"
        assert cache_path is not None
        return pd.DataFrame(
            {
                "submitter_id": ["TCGA-DX-0001", "TCGA-DX-0002", "TCGA-DX-0003"],
                "primary_diagnosis": [
                    "Liposarcoma, well differentiated",
                    "Dedifferentiated liposarcoma",
                    "Dedifferentiated liposarcoma",
                ],
            }
        )

    monkeypatch.setattr(
        expression_builders,
        "treehouse_gdc_primary_diagnosis_map",
        fake_diagnosis_map,
    )
    result = expression_builders.build_treehouse_source_matrices(source, cache_dir=tmp_path)

    assert set(result.matrix_paths) == {"SARC_WDLPS", "SARC_DDLPS"}
    wdlps = pd.read_parquet(result.matrix_paths["SARC_WDLPS"]).set_index("Symbol")
    ddlps = pd.read_parquet(result.matrix_paths["SARC_DDLPS"]).set_index("Symbol")
    assert list(wdlps.columns) == ["Ensembl_Gene_ID", "TCGA-DX-0001-01A"]
    assert list(ddlps.columns) == [
        "Ensembl_Gene_ID",
        "TCGA-DX-0002-01A",
        "TCGA-DX-0003-01A",
    ]
    np.testing.assert_allclose(wdlps.loc["TP53", "TCGA-DX-0001-01A"], 2.0)
    np.testing.assert_allclose(ddlps.loc["MDM2", "TCGA-DX-0002-01A"], 8.0)


def test_build_treehouse_source_matrices_splits_cbio_clinical_cohorts(
    tmp_path,
    monkeypatch,
):
    clinical_path = tmp_path / "clinical.tsv"
    pd.DataFrame(
        [
            {"th_dataset_id": "TCGA-BR-0001-01A", "disease": "breast invasive carcinoma"},
            {"th_dataset_id": "TCGA-BR-0002-01A", "disease": "breast invasive carcinoma"},
            {"th_dataset_id": "TREEHOUSE-3", "disease": "breast invasive carcinoma"},
        ]
    ).to_csv(clinical_path, sep="\t", index=False)
    tpm_path = tmp_path / "treehouse.tsv"
    pd.DataFrame(
        {
            "Gene": ["TP53", "ERBB2"],
            "TCGA-BR-0001-01A": np.log2(np.array([2.0, 1.0]) + 1.0),
            "TCGA-BR-0002-01A": np.log2(np.array([4.0, 8.0]) + 1.0),
            "TREEHOUSE-3": np.log2(np.array([100.0, 100.0]) + 1.0),
        }
    ).to_csv(tpm_path, sep="\t", index=False)
    source = expression_builders.TreehouseSource(
        source_id="synthetic-treehouse-brca-pam50",
        source_cohort="SYNTHETIC_TREEHOUSE_BRCA_PAM50",
        cancer_code=["BRCA_Basal", "BRCA_HER2"],
        tpm_file=tpm_path.name,
        clinical_file=clinical_path.name,
        source_project="Treehouse (TCGA-BRCA) x cBioPortal PAM50",
        cohorts=(
            expression_builders.TreehouseCohort(
                "BRCA_Basal",
                "breast invasive carcinoma",
                selection="cbio_clinical:brca_tcga_pan_can_atlas_2018:SUBTYPE:BRCA_Basal",
                cache_stem="tcga_brca_basal",
            ),
            expression_builders.TreehouseCohort(
                "BRCA_HER2",
                "breast invasive carcinoma",
                selection="cbio_clinical:brca_tcga_pan_can_atlas_2018:SUBTYPE:BRCA_Her2",
                cache_stem="tcga_brca_her2",
            ),
        ),
    )

    def fake_clinical_map(
        study_id,
        attribute_id,
        *,
        clinical_data_type="PATIENT",
        cache_path=None,
        force_download=False,
    ):
        assert study_id == "brca_tcga_pan_can_atlas_2018"
        assert attribute_id == "SUBTYPE"
        assert clinical_data_type == "PATIENT"
        assert cache_path is not None
        return pd.DataFrame(
            {
                "case_id": ["TCGA-BR-0001", "TCGA-BR-0002"],
                "value": ["BRCA_Basal", "BRCA_Her2"],
            }
        )

    monkeypatch.setattr(
        expression_builders,
        "treehouse_cbioportal_clinical_attribute_map",
        fake_clinical_map,
    )
    result = expression_builders.build_treehouse_source_matrices(source, cache_dir=tmp_path)

    assert set(result.matrix_paths) == {"BRCA_Basal", "BRCA_HER2"}
    basal = pd.read_parquet(result.matrix_paths["BRCA_Basal"]).set_index("Symbol")
    her2 = pd.read_parquet(result.matrix_paths["BRCA_HER2"]).set_index("Symbol")
    assert list(basal.columns) == ["Ensembl_Gene_ID", "TCGA-BR-0001-01A"]
    assert list(her2.columns) == ["Ensembl_Gene_ID", "TCGA-BR-0002-01A"]
    np.testing.assert_allclose(basal.loc["TP53", "TCGA-BR-0001-01A"], 2.0)
    np.testing.assert_allclose(her2.loc["ERBB2", "TCGA-BR-0002-01A"], 8.0)


def test_build_treehouse_source_matrices_splits_cbio_sample_numeric_cohorts(
    tmp_path,
    monkeypatch,
):
    clinical_path = tmp_path / "clinical.tsv"
    pd.DataFrame(
        [
            {"th_dataset_id": "TCGA-CO-0001-01A", "disease": "colon adenocarcinoma"},
            {"th_dataset_id": "TCGA-CO-0002-01A", "disease": "colon adenocarcinoma"},
            {"th_dataset_id": "TCGA-RE-0003-01A", "disease": "rectum adenocarcinoma"},
        ]
    ).to_csv(clinical_path, sep="\t", index=False)
    tpm_path = tmp_path / "treehouse.tsv"
    pd.DataFrame(
        {
            "Gene": ["TP53", "MLH1"],
            "TCGA-CO-0001-01A": np.log2(np.array([2.0, 1.0]) + 1.0),
            "TCGA-CO-0002-01A": np.log2(np.array([4.0, 8.0]) + 1.0),
            "TCGA-RE-0003-01A": np.log2(np.array([6.0, 3.0]) + 1.0),
        }
    ).to_csv(tpm_path, sep="\t", index=False)
    source = expression_builders.TreehouseSource(
        source_id="synthetic-treehouse-coadread-msi",
        source_cohort="SYNTHETIC_TREEHOUSE_COADREAD_MSI",
        cancer_code=["COAD_MSI", "COAD_MSS"],
        tpm_file=tpm_path.name,
        clinical_file=clinical_path.name,
        source_project="Treehouse (TCGA-COAD/READ) x cBioPortal MSIsensor",
        cohorts=(
            expression_builders.TreehouseCohort(
                "COAD_MSI",
                "colon adenocarcinoma",
                selection="cbio_sample_clinical:coadread_tcga_pan_can_atlas_2018:MSI_SENSOR_SCORE:>=10",
                cache_stem="tcga_coad_msi",
            ),
            expression_builders.TreehouseCohort(
                "COAD_MSS",
                "colon adenocarcinoma",
                selection="cbio_sample_clinical:coadread_tcga_pan_can_atlas_2018:MSI_SENSOR_SCORE:<10",
                cache_stem="tcga_coad_mss",
            ),
        ),
    )

    def fake_clinical_map(
        study_id,
        attribute_id,
        *,
        clinical_data_type="PATIENT",
        cache_path=None,
        force_download=False,
    ):
        assert study_id == "coadread_tcga_pan_can_atlas_2018"
        assert attribute_id == "MSI_SENSOR_SCORE"
        assert clinical_data_type == "SAMPLE"
        assert cache_path is not None
        return pd.DataFrame(
            {
                "case_id": ["TCGA-CO-0001", "TCGA-CO-0002", "TCGA-RE-0003"],
                "value": ["12.5", "2.0", "14.0"],
            }
        )

    monkeypatch.setattr(
        expression_builders,
        "treehouse_cbioportal_clinical_attribute_map",
        fake_clinical_map,
    )
    result = expression_builders.build_treehouse_source_matrices(source, cache_dir=tmp_path)

    assert set(result.matrix_paths) == {"COAD_MSI", "COAD_MSS"}
    msi = pd.read_parquet(result.matrix_paths["COAD_MSI"]).set_index("Symbol")
    mss = pd.read_parquet(result.matrix_paths["COAD_MSS"]).set_index("Symbol")
    assert list(msi.columns) == ["Ensembl_Gene_ID", "TCGA-CO-0001-01A"]
    assert list(mss.columns) == ["Ensembl_Gene_ID", "TCGA-CO-0002-01A"]
    np.testing.assert_allclose(msi.loc["TP53", "TCGA-CO-0001-01A"], 2.0)
    np.testing.assert_allclose(mss.loc["MLH1", "TCGA-CO-0002-01A"], 8.0)


def test_build_treehouse_source_matrices_splits_cbio_mutation_cohorts(
    tmp_path,
    monkeypatch,
):
    clinical_path = tmp_path / "clinical.tsv"
    pd.DataFrame(
        [
            {"th_dataset_id": "TCGA-LU-0001-01A", "disease": "lung adenocarcinoma"},
            {"th_dataset_id": "TCGA-LU-0002-01A", "disease": "lung adenocarcinoma"},
            {"th_dataset_id": "TCGA-LU-0003-01A", "disease": "lung adenocarcinoma"},
        ]
    ).to_csv(clinical_path, sep="\t", index=False)
    tpm_path = tmp_path / "treehouse.tsv"
    pd.DataFrame(
        {
            "Gene": ["TP53", "EGFR"],
            "TCGA-LU-0001-01A": np.log2(np.array([2.0, 1.0]) + 1.0),
            "TCGA-LU-0002-01A": np.log2(np.array([4.0, 8.0]) + 1.0),
            "TCGA-LU-0003-01A": np.log2(np.array([6.0, 3.0]) + 1.0),
        }
    ).to_csv(tpm_path, sep="\t", index=False)
    source = expression_builders.TreehouseSource(
        source_id="synthetic-treehouse-luad-mut",
        source_cohort="SYNTHETIC_TREEHOUSE_LUAD_MUT",
        cancer_code=["LUAD_EGFR", "LUAD_STK11"],
        tpm_file=tpm_path.name,
        clinical_file=clinical_path.name,
        source_project="Treehouse (TCGA-LUAD) x cBioPortal mutation calls",
        cohorts=(
            expression_builders.TreehouseCohort(
                "LUAD_EGFR",
                "lung adenocarcinoma",
                selection="cbio_mutation:luad_tcga_pan_can_atlas_2018:EGFR",
                cache_stem="tcga_luad_egfr",
            ),
            expression_builders.TreehouseCohort(
                "LUAD_STK11",
                "lung adenocarcinoma",
                selection="cbio_mutation:luad_tcga_pan_can_atlas_2018:STK11,KEAP1",
                cache_stem="tcga_luad_stk11",
            ),
        ),
    )

    def fake_mutation_case_set(
        study_id,
        gene_symbols,
        *,
        molecular_profile_id=None,
        sample_list_id=None,
        cache_path=None,
        force_download=False,
    ):
        assert study_id == "luad_tcga_pan_can_atlas_2018"
        assert cache_path is not None
        genes = tuple(gene_symbols)
        if genes == ("EGFR",):
            return pd.DataFrame(
                {
                    "case_id": ["TCGA-LU-0001"],
                    "sample_id": ["TCGA-LU-0001-01"],
                    "gene_symbol": ["EGFR"],
                    "entrez_gene_id": [1956],
                }
            )
        assert genes == ("STK11", "KEAP1")
        return pd.DataFrame(
            {
                "case_id": ["TCGA-LU-0002", "TCGA-LU-0003"],
                "sample_id": ["TCGA-LU-0002-01", "TCGA-LU-0003-01"],
                "gene_symbol": ["STK11", "KEAP1"],
                "entrez_gene_id": [6794, 9817],
            }
        )

    monkeypatch.setattr(
        expression_builders,
        "treehouse_cbioportal_mutation_case_set",
        fake_mutation_case_set,
    )
    result = expression_builders.build_treehouse_source_matrices(source, cache_dir=tmp_path)

    assert set(result.matrix_paths) == {"LUAD_EGFR", "LUAD_STK11"}
    egfr = pd.read_parquet(result.matrix_paths["LUAD_EGFR"]).set_index("Symbol")
    stk11 = pd.read_parquet(result.matrix_paths["LUAD_STK11"]).set_index("Symbol")
    assert list(egfr.columns) == ["Ensembl_Gene_ID", "TCGA-LU-0001-01A"]
    assert list(stk11.columns) == [
        "Ensembl_Gene_ID",
        "TCGA-LU-0002-01A",
        "TCGA-LU-0003-01A",
    ]
    np.testing.assert_allclose(egfr.loc["TP53", "TCGA-LU-0001-01A"], 2.0)
    np.testing.assert_allclose(stk11.loc["EGFR", "TCGA-LU-0002-01A"], 8.0)


def test_recount3_gene_sums_to_tpm_length_normalizes_and_collapses_versions():
    gene_sums = pd.DataFrame(
        {"S1": [1000.0, 1000.0], "S2": [0.0, 500.0]},
        index=["ENSG00000000001.5", "ENSG00000000002.3"],
    )
    bp_length = pd.Series({"ENSG00000000001": 1000.0, "ENSG00000000002": 2000.0})

    tpm = expression_builders.recount3_gene_sums_to_tpm(gene_sums, bp_length)

    np.testing.assert_allclose(tpm.sum(axis=0).to_numpy(), [1e6, 1e6])
    np.testing.assert_allclose(tpm["S1"].to_numpy(), [2e6 / 3, 1e6 / 3], rtol=1e-9)
    np.testing.assert_allclose(tpm["S2"].to_numpy(), [0.0, 1e6])

    dup = pd.DataFrame(
        {"S1": [600.0, 400.0]},
        index=["ENSG00000000003.1", "ENSG00000000003.1_PAR_Y"],
    )
    collapsed = expression_builders.recount3_gene_sums_to_tpm(
        dup,
        pd.Series({"ENSG00000000003": 1000.0}),
    )
    assert collapsed.index.tolist() == ["ENSG00000000003"]
    np.testing.assert_allclose(collapsed["S1"].to_numpy(), [1e6])


def test_recount3_parse_attributes_and_aggregate_runs_to_samples():
    attrs = expression_builders.parse_recount3_sample_attributes(
        "origin;;pancreas|type;;liver metastasis|n;;1"
    )
    assert attrs == {"origin": "pancreas", "type": "liver metastasis", "n": "1"}
    assert expression_builders.parse_recount3_sample_attributes("") == {}

    gene_sums = pd.DataFrame(
        {"R1": [10.0, 1.0], "R2": [20.0, 3.0], "R3": [5.0, 5.0], "R4": [99.0, 99.0]},
        index=["g1", "g2"],
    )
    meta = pd.DataFrame(
        {
            "external_id": ["R1", "R2", "R3", "R4"],
            "sample_acc": ["A", "A", "B", "C"],
        }
    )

    sample_gs, sample_meta = expression_builders.aggregate_recount3_runs_to_samples(
        gene_sums,
        meta,
        keep_runs={"R1", "R2", "R3"},
    )

    assert set(sample_gs.columns) == {"A", "B"}
    np.testing.assert_allclose(sample_gs.loc["g1", "A"], 30.0)
    np.testing.assert_allclose(sample_gs.loc["g2", "A"], 4.0)
    np.testing.assert_allclose(sample_gs.loc["g1", "B"], 5.0)
    assert list(sample_meta.index) == list(sample_gs.columns)


def test_build_recount3_source_matrices_writes_canonical_artifacts(tmp_path, monkeypatch):
    annotation = pd.DataFrame(
        {
            "bp_length": [1000.0, 2000.0],
            "Symbol": ["TP53", "EGFR"],
        },
        index=["ENSG00000141510", "ENSG00000146648"],
    )
    gene_sums = pd.DataFrame(
        {
            "R1": [100.0, 50.0],
            "R2": [50.0, 25.0],
            "R3": [0.0, 80.0],
            "R4": [999.0, 999.0],
        },
        index=["ENSG00000141510.1", "ENSG00000146648.2"],
    )
    metadata = pd.DataFrame(
        {
            "external_id": ["R1", "R2", "R3", "R4"],
            "sample_acc": ["SAMPLE_A", "SAMPLE_A", "SAMPLE_B", "CONTROL"],
            "sample_attributes": [
                "code;;CODE_A",
                "code;;CODE_A",
                "code;;CODE_B",
                "code;;CONTROL",
            ],
            "sample_title": ["A1", "A2", "B1", "C1"],
        }
    )
    monkeypatch.setattr(
        expression_builders,
        "fetch_recount3_gene_annotation",
        lambda _cache: annotation,
    )
    monkeypatch.setattr(
        expression_builders,
        "fetch_recount3_gene_sums",
        lambda _srp, _cache: gene_sums,
    )
    monkeypatch.setattr(
        expression_builders,
        "fetch_recount3_sample_metadata",
        lambda _srp, _cache: metadata,
    )
    source = expression_builders.Recount3Source(
        source_id="synthetic-recount3",
        srp="SRP000000",
        source_cohort="SYNTHETIC_RECOUNT3",
        cancer_code=["CODE_A", "CODE_B"],
        sample_to_cancer_code=lambda attrs, _title: (
            attrs.get("code") if attrs.get("code") != "CONTROL" else None
        ),
        expected_n={"CODE_A": 1, "CODE_B": 1},
    )
    out_dir = tmp_path / "derived"
    out_dir.mkdir()
    stale_matrix = out_dir / "STALE_per_sample_tpm.parquet"
    stale_qc = out_dir / "STALE_sample_qc.csv"
    stale_matrix.write_text("stale matrix")
    stale_qc.write_text("stale qc")

    result = expression_builders.build_recount3_source_matrices(source, cache_dir=tmp_path)

    assert set(result.matrix_paths) == {"CODE_A", "CODE_B"}
    assert not stale_matrix.exists()
    assert not stale_qc.exists()
    assert result.sidecar_paths["mapping_audit"].exists()
    assert result.sidecar_paths["parse_diagnostics"].exists()
    code_a = pd.read_parquet(result.matrix_paths["CODE_A"])
    code_b = pd.read_parquet(result.matrix_paths["CODE_B"])
    assert list(code_a.columns) == ["Ensembl_Gene_ID", "Symbol", "SAMPLE_A"]
    assert list(code_b.columns) == ["Ensembl_Gene_ID", "Symbol", "SAMPLE_B"]
    assert set(code_a["Ensembl_Gene_ID"]) == {"ENSG00000141510", "ENSG00000146648"}
    assert np.isclose(code_a["SAMPLE_A"].sum(), 1_000_000.0)
    assert np.isclose(code_b["SAMPLE_B"].sum(), 1_000_000.0)
    assert set(result.sample_qc["cancer_code"]) == {"CODE_A", "CODE_B"}


def test_ensembl_transcript_gene_map_reads_versioned_fasta_headers(tmp_path):
    fasta = tmp_path / "ensembl.fa"
    fasta.write_text(
        ">ENST000001.2 cdna chromosome:GRCh38:1:1:2:1 gene:ENSG000001.7\n"
        "ACGT\n"
        ">ENST000002.1 cdna chromosome:GRCh38:1:3:4:1 gene:ENSG000002.3\n"
        "TGCA\n"
    )

    assert expression_builders.ensembl_transcript_gene_map(fasta) == {
        "ENST000001.2": "ENSG000001.7",
        "ENST000002.1": "ENSG000002.3",
    }


def test_prepare_salmon_transcriptome_combines_all_pinned_fastas(tmp_path):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    cdna = reference_dir / "cdna.fa.gz"
    ncrna = reference_dir / "ncrna.fa.gz"
    contents = (
        b">ENST000001.1 gene:ENSG000001.1\nACGT\n",
        b">ENST000002.1 gene:ENSG000002.1\nTGCA\n",
    )
    for path, content in zip((cdna, ncrna), contents):
        with (
            path.open("wb") as raw,
            gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                mtime=0,
            ) as handle,
        ):
            handle.write(content)

    expected = tmp_path / "expected.fa.gz"
    with (
        expected.open("wb") as raw,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=6,
            mtime=0,
        ) as handle,
    ):
        for content in contents:
            handle.write(content)
    source = replace(
        _synthetic_sra_salmon_source(),
        reference_fastas=(
            expression_builders.SalmonReferenceFasta(
                url="https://example.org/cdna.fa.gz",
                file_name=cdna.name,
                sha256=expression_builders._sha256(cdna),
            ),
            expression_builders.SalmonReferenceFasta(
                url="https://example.org/ncrna.fa.gz",
                file_name=ncrna.name,
                sha256=expression_builders._sha256(ncrna),
            ),
        ),
        combined_transcriptome_file="combined.fa.gz",
        combined_transcriptome_sha256=expression_builders._sha256(expected),
    )

    combined = expression_builders.prepare_salmon_transcriptome(source, cache_dir=tmp_path)

    assert expression_builders._sha256(combined) == expression_builders._sha256(expected)
    assert expression_builders.ensembl_transcript_gene_map(combined) == {
        "ENST000001.1": "ENSG000001.1",
        "ENST000002.1": "ENSG000002.1",
    }


def test_prepare_salmon_index_uses_supplied_transcriptome_without_download(
    tmp_path,
    monkeypatch,
):
    transcriptome = tmp_path / "supplied.fa.gz"
    transcriptome.write_bytes(b"reviewed combined transcriptome")
    source = replace(
        _synthetic_sra_salmon_source(),
        combined_transcriptome_sha256=expression_builders._sha256(transcriptome),
    )
    commands = []
    monkeypatch.setattr(
        expression_builders,
        "prepare_salmon_transcriptome",
        lambda *_args, **_kwargs: pytest.fail("supplied transcriptome should prevent download"),
    )
    monkeypatch.setattr(expression_builders, "_salmon_executable", lambda _command: "salmon")
    monkeypatch.setattr(expression_builders, "_salmon_version", lambda _executable: "1.10.3")

    def _run(command, check):
        commands.append((command, check))
        output = Path(command[command.index("--index") + 1])
        output.mkdir(parents=True)
        (output / "versionInfo.json").write_text("{}")

    monkeypatch.setattr(expression_builders.subprocess, "run", _run)
    selected, index_dir = expression_builders.prepare_salmon_index(
        source,
        cache_dir=tmp_path / "cache",
        transcriptome_path=transcriptome,
        force_download=True,
    )

    assert selected == transcriptome
    assert index_dir.exists()
    assert (index_dir / "oncoref_salmon_version.txt").read_text() == "1.10.3\n"
    assert commands[0][0][commands[0][0].index("--transcripts") + 1] == str(transcriptome)


@pytest.mark.parametrize(
    ("cached_version", "expected_rebuild"),
    [("1.10.3", False), ("1.10.2", True)],
)
def test_prepare_salmon_index_cache_is_keyed_by_salmon_version(
    tmp_path,
    monkeypatch,
    cached_version,
    expected_rebuild,
):
    transcriptome = tmp_path / "supplied.fa.gz"
    transcriptome.write_bytes(b"reviewed combined transcriptome")
    source = replace(
        _synthetic_sra_salmon_source(),
        combined_transcriptome_sha256=expression_builders._sha256(transcriptome),
    )
    index_dir = tmp_path / "cache" / "reference" / "salmon_index"
    index_dir.mkdir(parents=True)
    (index_dir / "versionInfo.json").write_text("{}")
    (index_dir / "oncoref_transcriptome_sha256.txt").write_text(
        source.combined_transcriptome_sha256 + "\n"
    )
    (index_dir / "oncoref_salmon_version.txt").write_text(cached_version + "\n")
    commands = []

    monkeypatch.setattr(expression_builders, "_salmon_executable", lambda _command: "salmon")
    monkeypatch.setattr(expression_builders, "_salmon_version", lambda _executable: "1.10.3")

    def _run(command, check):
        commands.append((command, check))
        output = Path(command[command.index("--index") + 1])
        output.mkdir(parents=True)
        (output / "versionInfo.json").write_text("{}")

    monkeypatch.setattr(expression_builders.subprocess, "run", _run)
    _, selected_index = expression_builders.prepare_salmon_index(
        source,
        cache_dir=tmp_path / "cache",
        transcriptome_path=transcriptome,
    )

    assert selected_index == index_dir
    assert bool(commands) is expected_rebuild
    assert (index_dir / "oncoref_salmon_version.txt").read_text() == "1.10.3\n"


def test_salmon_version_reads_resolved_executable_output(monkeypatch):
    calls = []

    def _run(command, *, check, capture_output, text):
        calls.append((command, check, capture_output, text))
        return subprocess.CompletedProcess(command, 0, stdout="salmon 1.10.3\n", stderr="")

    monkeypatch.setattr(expression_builders.subprocess, "run", _run)

    assert expression_builders._salmon_version("/opt/salmon") == "1.10.3"
    assert calls == [(["/opt/salmon", "--version"], True, True, True)]


def test_quantify_sra_salmon_run_reuses_an_exact_input_cache(tmp_path, monkeypatch):
    source = _synthetic_sra_salmon_source()
    run = source.runs[0]
    quant_dir = tmp_path / "quant" / run.accession
    quant_dir.mkdir(parents=True)
    (quant_dir / "quant.sf").write_text("cached")
    (quant_dir / "oncoref_quantification_inputs.json").write_text(
        json.dumps(
            expression_builders._sra_salmon_quantification_inputs(
                source,
                run,
                salmon_version="1.10.3",
            )
        )
    )
    monkeypatch.setattr(expression_builders, "_salmon_executable", lambda _command: "salmon")
    monkeypatch.setattr(expression_builders, "_salmon_version", lambda _executable: "1.10.3")
    monkeypatch.setattr(
        expression_builders,
        "download_sra_salmon_run_reads",
        lambda *_args, **_kwargs: pytest.fail("exact cache should not download reads"),
    )

    result = expression_builders.quantify_sra_salmon_run(
        source,
        run,
        cache_dir=tmp_path,
        index_dir=tmp_path / "index",
    )

    assert result.read_text() == "cached"


def test_quantify_sra_salmon_run_forces_read_refresh_on_cache_hit(tmp_path, monkeypatch):
    source = _synthetic_sra_salmon_source()
    run = source.runs[0]
    quant_dir = tmp_path / "quant" / run.accession
    quant_dir.mkdir(parents=True)
    (quant_dir / "quant.sf").write_text("cached")
    (quant_dir / "oncoref_quantification_inputs.json").write_text(
        json.dumps(
            expression_builders._sra_salmon_quantification_inputs(
                source,
                run,
                salmon_version="1.10.3",
            )
        )
    )
    refresh_calls = []

    def _refresh(refreshed_run, cache_dir, *, force_download):
        refresh_calls.append((refreshed_run, Path(cache_dir), force_download))
        return tmp_path / "read_1.fastq.gz", tmp_path / "read_2.fastq.gz"

    monkeypatch.setattr(expression_builders, "download_sra_salmon_run_reads", _refresh)
    monkeypatch.setattr(
        expression_builders,
        "_salmon_executable",
        lambda _command: "salmon",
    )
    monkeypatch.setattr(expression_builders, "_salmon_version", lambda _executable: "1.10.3")
    result = expression_builders.quantify_sra_salmon_run(
        source,
        run,
        cache_dir=tmp_path,
        index_dir=tmp_path / "index",
        force_download=True,
    )

    assert result.read_text() == "cached"
    assert refresh_calls == [(run, tmp_path, True)]


def test_external_salmon_quant_paths_still_honor_forced_read_downloads(
    tmp_path,
    monkeypatch,
):
    source = _synthetic_sra_salmon_source()
    quant_paths = {
        run.accession: tmp_path / "external" / run.accession / "quant.sf" for run in source.runs
    }
    refreshed = []

    def _refresh(run, cache_dir, *, force_download):
        refreshed.append((run.accession, Path(cache_dir), force_download))
        return tmp_path / "read_1.fastq.gz", tmp_path / "read_2.fastq.gz"

    monkeypatch.setattr(expression_builders, "download_sra_salmon_run_reads", _refresh)
    resolved, transcriptome = expression_builders._resolve_sra_salmon_quant_paths(
        source,
        cache_dir=tmp_path / "cache",
        quant_paths=quant_paths,
        transcriptome_path=None,
        salmon_executable="salmon",
        threads=4,
        force_download=True,
        force_index=False,
        force_quant=False,
    )

    assert resolved == quant_paths
    assert transcriptome is None
    assert refreshed == [(run.accession, tmp_path / "cache", True) for run in source.runs]


@pytest.mark.parametrize("force_flag", ["force_index", "force_quant"])
def test_external_salmon_quant_paths_reject_regeneration_flags(tmp_path, force_flag):
    source = _synthetic_sra_salmon_source()
    options = {
        "source": source,
        "cache_dir": tmp_path / "cache",
        "quant_paths": {
            run.accession: tmp_path / "external" / run.accession / "quant.sf" for run in source.runs
        },
        "transcriptome_path": None,
        "salmon_executable": "salmon",
        "threads": 4,
        "force_download": False,
        "force_index": False,
        "force_quant": False,
    }
    options[force_flag] = True

    with pytest.raises(ValueError, match=force_flag):
        expression_builders._resolve_sra_salmon_quant_paths(**options)


@pytest.mark.parametrize(
    "changed_input",
    ["transcriptome", "read_md5", "salmon_args", "salmon_version"],
)
def test_quantify_sra_salmon_run_rebuilds_when_cache_inputs_change(
    tmp_path,
    monkeypatch,
    changed_input,
):
    original_source = _synthetic_sra_salmon_source()
    original_run = original_source.runs[0]
    source = original_source
    run = original_run
    if changed_input == "transcriptome":
        source = replace(source, combined_transcriptome_sha256="e" * 64)
    elif changed_input == "read_md5":
        run = replace(run, read_md5=("e" * 32, run.read_md5[1]))
        source = replace(source, runs=(run, *source.runs[1:]))
    elif changed_input == "salmon_args":
        source = replace(source, salmon_args=(*source.salmon_args, "--posBias"))

    quant_dir = tmp_path / "quant" / run.accession
    quant_dir.mkdir(parents=True)
    (quant_dir / "quant.sf").write_text("stale")
    (quant_dir / "oncoref_quantification_inputs.json").write_text(
        json.dumps(
            expression_builders._sra_salmon_quantification_inputs(
                original_source,
                original_run,
                salmon_version="1.10.2" if changed_input == "salmon_version" else "1.10.3",
            )
        )
    )
    read_1 = tmp_path / "read_1.fastq.gz"
    read_2 = tmp_path / "read_2.fastq.gz"
    read_1.write_bytes(b"reads")
    read_2.write_bytes(b"reads")
    commands = []

    monkeypatch.setattr(
        expression_builders,
        "download_sra_salmon_run_reads",
        lambda *_args, **_kwargs: (read_1, read_2),
    )
    monkeypatch.setattr(expression_builders, "_salmon_executable", lambda _command: "salmon")
    monkeypatch.setattr(expression_builders, "_salmon_version", lambda _executable: "1.10.3")

    def _run(command, check):
        commands.append((command, check))
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "quant.sf").write_text("fresh")

    monkeypatch.setattr(expression_builders.subprocess, "run", _run)
    result = expression_builders.quantify_sra_salmon_run(
        source,
        run,
        cache_dir=tmp_path,
        index_dir=tmp_path / "index",
    )

    assert result.read_text() == "fresh"
    assert commands
    assert json.loads((quant_dir / "oncoref_quantification_inputs.json").read_text()) == (
        expression_builders._sra_salmon_quantification_inputs(
            source,
            run,
            salmon_version="1.10.3",
        )
    )


def test_salmon_gene_tpm_matrix_rejects_unmapped_expression(tmp_path):
    quant_path = tmp_path / "quant.sf"
    pd.DataFrame(
        {
            "Name": ["ENST_MAPPED", "ENST_UNMAPPED"],
            "Length": [1000, 1000],
            "EffectiveLength": [800, 800],
            "TPM": [999_999.0, 1.0],
            "NumReads": [100, 1],
        }
    ).to_csv(quant_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="absent from the pinned transcript-to-gene map"):
        expression_builders.salmon_gene_tpm_matrix(
            {"SRR00000001": quant_path},
            {"ENST_MAPPED": "ENSG00000141510.1"},
        )


def test_ncbi_gene_lengths_use_exon_union_gene_fallback_and_primary_assembly(tmp_path):
    annotation = tmp_path / "annotation.gff.gz"
    assembly_report = _write_synthetic_ncbi_assembly_report(tmp_path)
    lines = [
        "##gff-version 3",
        (
            "NC_000017.11\tRefSeq\tregion\t1\t83257441\t.\t+\t.\t"
            "ID=NC_000017.11:1..83257441;Name=17;chromosome=17;genome=chromosome"
        ),
        (
            "NC_000017.11\tRefSeq\tgene\t100\t300\t.\t+\t.\t"
            "ID=gene-TP53;Dbxref=GeneID:7157;Name=TP53;gene=TP53"
        ),
        ("NC_000017.11\tRefSeq\texon\t100\t199\t.\t+\t.\tID=exon-1;Dbxref=GeneID:7157;gene=TP53"),
        ("NC_000017.11\tRefSeq\texon\t150\t249\t.\t+\t.\tID=exon-2;Dbxref=GeneID:7157;gene=TP53"),
        (
            "NW_012132914.1\tRefSeq\tregion\t1\t467143\t.\t+\t.\t"
            "ID=NW_012132914.1:1..467143;Name=1;genome=genomic;map=1p36.21"
        ),
        (
            "NW_012132914.1\tRefSeq\texon\t1\t10000\t.\t+\t.\t"
            "ID=alt-exon;Dbxref=GeneID:7157;gene=TP53"
        ),
        (
            "NT_187361.1\tRefSeq\tregion\t1\t175055\t.\t+\t.\t"
            "ID=NT_187361.1:1..175055;Name=1;genome=genomic;map=unlocalized"
        ),
        (
            "NT_187361.1\tRefSeq\tgene\t10\t109\t.\t+\t.\t"
            "ID=gene-LOC105379854;Dbxref=GeneID:105379854;Name=LOC105379854;"
            "gene=LOC105379854"
        ),
        (
            "NC_000007.14\tRefSeq\tregion\t1\t159345973\t.\t+\t.\t"
            "ID=NC_000007.14:1..159345973;Name=7;chromosome=7;genome=chromosome"
        ),
        (
            "NC_000007.14\tRefSeq\tgene\t1000\t3000\t.\t+\t.\t"
            "ID=gene-EGFR;Dbxref=GeneID:1956;Name=EGFR;gene=EGFR"
        ),
    ]
    with gzip.open(annotation, "wt") as handle:
        handle.write("\n".join(lines) + "\n")

    primary_sequences = expression_builders.ncbi_primary_assembly_sequences(assembly_report)
    assert primary_sequences == {"NC_000017.11", "NC_000007.14", "NT_187361.1"}

    lengths = expression_builders.ncbi_gene_lengths_from_gff(
        annotation,
        gene_ids=["7157", "1956", "105379854", "999999999"],
        primary_sequences=primary_sequences,
    ).set_index("Geneid")

    assert lengths.loc["7157", "gene_length_bp"] == 150
    assert lengths.loc["7157", "length_source"] == "exon_union"
    assert lengths.loc["1956", "gene_length_bp"] == 2001
    assert lengths.loc["1956", "length_source"] == "gene_span"
    assert lengths.loc["105379854", "gene_length_bp"] == 100
    assert pd.isna(lengths.loc["999999999", "gene_length_bp"])
    assert lengths.loc["999999999", "length_source"] == "unresolved"


def test_sra_ncbi_count_matrix_requires_exact_run_manifest_and_gene_order(tmp_path):
    runs = tuple(
        expression_builders.SraNcbiCountRun(
            accession=accession,
            biosample=f"SAMN{accession[-8:]}",
            sample_title=accession,
            role="tumor",
            analysis_accession=f"SRZ{accession[-8:]}",
            counts_md5="a" * 32,
            cancer_code="SARC_MMNST",
        )
        for accession in ("SRR00000001", "SRR00000002")
    )
    source = expression_builders.SraNcbiCountSource(
        source_id="synthetic-ncbi-counts",
        bioproject="PRJNA000000",
        sra_study="SRP000000",
        source_cohort="SYNTHETIC_NCBI_COUNTS",
        cancer_code="SARC_MMNST",
        runs=runs,
        annotation_url="https://example.org/annotation.gff.gz",
        annotation_sha256="b" * 64,
        annotation_release="synthetic",
        assembly_report_url="https://example.org/assembly_report.txt",
        assembly_report_sha256="c" * 64,
        expected_gene_rows=2,
    )
    paths = {}
    for run, gene_ids in zip(runs, (["7157", "1956"], ["1956", "7157"])):
        path = tmp_path / f"{run.accession}.tsv"
        pd.DataFrame({"Geneid": gene_ids, f"{run.accession}_count": [10.0, 20.0]}).to_csv(
            path, sep="\t", index=False
        )
        paths[run.accession] = path

    with pytest.raises(ValueError, match="GeneID rows differ"):
        expression_builders.read_sra_ncbi_count_matrix(source, paths)

    paths["SRR99999999"] = paths[runs[0].accession]
    with pytest.raises(ValueError, match=r"unexpected=\['SRR99999999'\]"):
        expression_builders.read_sra_ncbi_count_matrix(source, paths)


def test_build_sra_ncbi_count_source_routes_only_tumor_and_audits_control(tmp_path):
    annotation = tmp_path / "annotation.gff.gz"
    assembly_report = _write_synthetic_ncbi_assembly_report(tmp_path)
    lines = [
        "##gff-version 3",
        (
            "NC_000017.11\tRefSeq\tregion\t1\t83257441\t.\t+\t.\t"
            "ID=NC_000017.11:1..83257441;Name=17;chromosome=17;genome=chromosome"
        ),
        (
            "NC_000017.11\tRefSeq\tgene\t100\t1099\t.\t+\t.\t"
            "ID=gene-TP53;Dbxref=GeneID:7157;Name=TP53;gene=TP53"
        ),
        (
            "NC_000017.11\tRefSeq\texon\t100\t1099\t.\t+\t.\t"
            "ID=exon-TP53;Dbxref=GeneID:7157;gene=TP53"
        ),
        (
            "NC_000007.14\tRefSeq\tregion\t1\t159345973\t.\t+\t.\t"
            "ID=NC_000007.14:1..159345973;Name=7;chromosome=7;genome=chromosome"
        ),
        (
            "NC_000007.14\tRefSeq\tgene\t2000\t3999\t.\t+\t.\t"
            "ID=gene-EGFR;Dbxref=GeneID:1956;Name=EGFR;gene=EGFR"
        ),
        (
            "NC_000007.14\tRefSeq\texon\t2000\t3999\t.\t+\t.\t"
            "ID=exon-EGFR;Dbxref=GeneID:1956;gene=EGFR"
        ),
    ]
    with gzip.open(annotation, "wt") as handle:
        handle.write("\n".join(lines) + "\n")

    count_paths = {}
    runs = []
    for accession, role, cancer_code, counts in (
        ("SRR00000001", "tumor", "SARC_MMNST", [100.0, 50.0]),
        ("SRR00000002", "normal_control", None, [20.0, 80.0]),
    ):
        count_path = tmp_path / f"{accession}.ncbi-gene-counts.tsv"
        pd.DataFrame(
            {
                "Geneid": ["7157", "1956"],
                f"{accession}_count": counts,
            }
        ).to_csv(count_path, sep="\t", index=False)
        count_paths[accession] = count_path
        runs.append(
            expression_builders.SraNcbiCountRun(
                accession=accession,
                biosample=f"SAMN{accession[-8:]}",
                sample_title=role,
                role=role,
                analysis_accession=f"SRZ{accession[-8:]}",
                counts_md5=hashlib.md5(count_path.read_bytes()).hexdigest(),
                cancer_code=cancer_code,
            )
        )

    source = expression_builders.SraNcbiCountSource(
        source_id="synthetic-ncbi-counts",
        bioproject="PRJNA000000",
        sra_study="SRP000000",
        source_cohort="SYNTHETIC_NCBI_COUNTS",
        cancer_code="SARC_MMNST",
        runs=tuple(runs),
        annotation_url="https://example.org/annotation.gff.gz",
        annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
        annotation_release="synthetic",
        assembly_report_url="https://example.org/assembly_report.txt",
        assembly_report_sha256=hashlib.sha256(assembly_report.read_bytes()).hexdigest(),
        expected_gene_rows=2,
        expected_n={"SARC_MMNST": 1},
    )
    with pytest.raises(ValueError, match=r"unexpected=\['SRR99999999'\]"):
        expression_builders.build_sra_ncbi_count_source_matrices(
            source,
            cache_dir=tmp_path / "rejected-cache",
            count_paths={**count_paths, "SRR99999999": count_paths["SRR00000001"]},
            annotation_path=annotation,
            assembly_report_path=assembly_report,
        )

    result = expression_builders.build_sra_ncbi_count_source_matrices(
        source,
        cache_dir=tmp_path / "cache",
        count_paths=count_paths,
        annotation_path=annotation,
        assembly_report_path=assembly_report,
    )

    matrix = result.matrices["SARC_MMNST"]
    assert list(matrix.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "SRR00000001",
    ]
    assert set(matrix["Symbol"]) == {"TP53", "EGFR"}
    assert matrix["SRR00000001"].sum() == pytest.approx(1_000_000.0)
    assert set(result.sample_qc["source_type"]) == {"sra-ncbi-counts"}
    manifest = pd.read_csv(result.sidecar_paths["run_manifest"])
    assert manifest["included"].tolist() == [True, False]
    assert manifest.loc[1, "exclusion_reason"] == (
        "independent_normal_control_excluded_from_tumor_reference"
    )
    coverage = pd.read_csv(result.sidecar_paths["ncbi_count_mapping_coverage"])
    assert coverage["reference_input_count_fraction"].tolist() == [1.0, 1.0]
    assert result.sidecar_paths["ncbi_gene_length_audit"].exists()


def test_build_sra_salmon_source_routes_only_tumors_and_audits_controls(tmp_path):
    source = _synthetic_sra_salmon_source()
    transcript_to_gene = {
        "ENST_TP53_A": "ENSG00000141510.17",
        "ENST_TP53_B": "ENSG00000141510.18",
        "ENST_EGFR": "ENSG00000146648.16",
    }
    quant_paths = {}
    for index, run in enumerate(source.runs):
        quant_dir = tmp_path / "quant" / run.accession
        quant_dir.mkdir(parents=True)
        quant_path = quant_dir / "quant.sf"
        tp53 = 600_000.0 - index * 10_000.0
        pd.DataFrame(
            {
                "Name": list(transcript_to_gene),
                "Length": [1000, 800, 2000],
                "EffectiveLength": [800, 600, 1800],
                "TPM": [tp53 / 2, tp53 / 2, 1_000_000.0 - tp53],
                "NumReads": [100, 100, 100],
            }
        ).to_csv(quant_path, sep="\t", index=False)
        aux_dir = quant_dir / "aux_info"
        aux_dir.mkdir()
        (aux_dir / "meta_info.json").write_text(
            json.dumps(
                {
                    "salmon_version": "1.10.3",
                    "library_types": ["ISR"],
                    "percent_mapped": 95.0 - index,
                    "num_processed": 1000,
                    "num_mapped": 950 - index * 10,
                }
            )
        )
        quant_paths[run.accession] = quant_path

    result = expression_builders.build_sra_salmon_source_matrices(
        source,
        cache_dir=tmp_path / "cache",
        quant_paths=quant_paths,
        transcript_to_gene=transcript_to_gene,
    )

    assert result.source is source
    assert set(result.matrices) == {"SARC_MMNST"}
    matrix = result.matrices["SARC_MMNST"]
    assert list(matrix.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "SRR00000000",
        "SRR00000001",
        "SRR00000002",
    ]
    assert set(matrix["Symbol"]) == {"TP53", "EGFR"}
    assert np.allclose(matrix.iloc[:, 2:].sum(axis=0), 1_000_000.0)
    assert set(result.sample_qc["sample_id"]) == {
        "SRR00000000",
        "SRR00000001",
        "SRR00000002",
    }
    assert set(result.sample_qc["source_type"]) == {"sra-salmon"}
    emitted_qc = pd.read_csv(result.sidecar_paths["SARC_MMNST_sample_qc"])
    assert set(emitted_qc["source_type"]) == {"sra-salmon"}
    assert set(result.summary_rows["n_samples"]) == {3}

    manifest = pd.read_csv(result.sidecar_paths["run_manifest"])
    quant_audit = pd.read_csv(result.sidecar_paths["salmon_quantification_audit"])
    assert len(manifest) == 6
    assert manifest["included"].value_counts().to_dict() == {True: 3, False: 3}
    assert len(quant_audit) == 6
    assert set(quant_audit["n_transcripts_unmapped"]) == {0}
    assert set(quant_audit["salmon_version"]) == {"1.10.3"}
    assert set(quant_audit["library_types"]) == {"ISR"}


def test_build_geo_matrix_script_uses_registry_config(tmp_path, capsys):
    source_path = tmp_path / "source.tsv"
    pd.DataFrame(
        {
            "gene": ["TP53", "EGFR"],
            "tumor_a": ["1", "3"],
            "normal_a": ["10", "10"],
        }
    ).to_csv(source_path, sep="\t", index=False)
    registry_path = tmp_path / "expression_sources.yaml"
    registry_path.write_text(
        """
sources:
  - id: synthetic-geo
    source_type: geo-matrix
    cancer_codes: [CODE_A]
    file_url: https://example.org/source.tsv.gz
    file_name: source.tsv.gz
    unit: TPM
    gene_id_col: gene
    source_cohort: SYNTHETIC_GEO
    source_project: GEO
    sample_filter:
      include_match: "^tumor_"
""".lstrip()
    )
    mod = _load_script("build_geo_matrix")

    status = mod.main(
        [
            "--source-id",
            "synthetic-geo",
            "--registry",
            str(registry_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-path",
            str(source_path),
        ]
    )

    assert status == 0
    out = pd.read_parquet(tmp_path / "out" / "CODE_A_per_sample_tpm.parquet")
    assert list(out.columns) == ["Ensembl_Gene_ID", "Symbol", "tumor_a"]
    assert set(out["Symbol"]) == {"TP53", "EGFR"}
    assert np.isclose(out["tumor_a"].sum(), 1_000_000.0)
    stdout = capsys.readouterr().out
    assert '"sample_counts": {' in stdout
    assert '"CODE_A": 1' in stdout


def test_build_recount3_script_uses_registry_config(tmp_path, monkeypatch, capsys):
    mod = _load_script("build_recount3_source")
    source = expression_builders.Recount3Source(
        source_id="synthetic-recount3",
        srp="SRP000000",
        source_cohort="SYNTHETIC_RECOUNT3",
        cancer_code="CODE_A",
    )

    def _fake_build(source_obj, *, cache_dir, output_dir=None, **_kwargs):
        assert source_obj is source
        assert Path(cache_dir) == tmp_path / "cache"
        assert output_dir is None
        matrix = pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["ENSG00000141510"],
                "Symbol": ["TP53"],
                "S1": [1_000_000.0],
            }
        )
        return expression_builders.SourceMatrixBuildResult(
            source=source,
            matrices={"CODE_A": matrix},
            matrix_paths={"CODE_A": tmp_path / "CODE_A_per_sample_tpm.parquet"},
            summary_rows=pd.DataFrame(
                columns=list(expression_builders.REFERENCE_EXPRESSION_COLUMNS)
            ),
            mapping_audit=pd.DataFrame(),
            parse_diagnostics=pd.DataFrame(),
            sample_qc=pd.DataFrame(),
            sidecar_paths={"mapping_audit": tmp_path / "mapping_audit.csv"},
        )

    monkeypatch.setattr(
        mod,
        "recount3_source_from_registry",
        lambda source_id, registry_path=None: source,
    )
    monkeypatch.setattr(mod, "build_recount3_source_matrices", _fake_build)

    assert mod.main(["synthetic-recount3", "--cache-dir", str(tmp_path / "cache")]) == 0
    stdout = capsys.readouterr().out
    assert '"source_id": "synthetic-recount3"' in stdout
    assert '"CODE_A": 1' in stdout


def test_build_sra_salmon_script_uses_registry_config(tmp_path, monkeypatch, capsys):
    mod = _load_script("build_sra_salmon_source")
    source = _synthetic_sra_salmon_source()

    def _fake_build(source_obj, *, cache_dir, quant_paths=None, **_kwargs):
        assert source_obj is source
        assert Path(cache_dir) == tmp_path / "cache"
        assert _kwargs["transcriptome_path"] == tmp_path / "supplied.fa.gz"
        assert _kwargs["force_download"] is True
        assert quant_paths == {
            run.accession: tmp_path / "quant" / run.accession / "quant.sf" for run in source.runs
        }
        matrix = pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["ENSG00000141510"],
                "Symbol": ["TP53"],
                "SRR00000000": [1_000_000.0],
            }
        )
        return expression_builders.SourceMatrixBuildResult(
            source=source,
            matrices={"SARC_MMNST": matrix},
            matrix_paths={"SARC_MMNST": tmp_path / "SARC_MMNST_per_sample_tpm.parquet"},
            summary_rows=pd.DataFrame(
                columns=list(expression_builders.REFERENCE_EXPRESSION_COLUMNS)
            ),
            mapping_audit=pd.DataFrame(),
            parse_diagnostics=pd.DataFrame(),
            sample_qc=pd.DataFrame(),
            sidecar_paths={"run_manifest": tmp_path / "run_manifest.csv"},
        )

    monkeypatch.setattr(
        mod,
        "sra_salmon_source_from_registry",
        lambda source_id, registry_path=None: source,
    )
    monkeypatch.setattr(mod, "build_sra_salmon_source_matrices", _fake_build)

    assert (
        mod.main(
            [
                "synthetic-sra-salmon",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--quant-dir",
                str(tmp_path / "quant"),
                "--transcriptome",
                str(tmp_path / "supplied.fa.gz"),
                "--force-download",
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert '"source_id": "synthetic-sra-salmon"' in stdout
    assert '"SARC_MMNST": 1' in stdout


@pytest.mark.parametrize("force_flag", ["--force-index", "--force-quant"])
def test_build_sra_salmon_script_rejects_external_regeneration(tmp_path, capsys, force_flag):
    mod = _load_script("build_sra_salmon_source")

    with pytest.raises(SystemExit) as error:
        mod.main(
            [
                "synthetic-sra-salmon",
                "--quant-dir",
                str(tmp_path / "quant"),
                force_flag,
            ]
        )

    assert error.value.code == 2
    assert "--quant-dir cannot be combined" in capsys.readouterr().err


def test_build_treehouse_script_uses_registry_config(tmp_path, monkeypatch, capsys):
    mod = _load_script("build_treehouse_source")
    source = expression_builders.TreehouseSource(
        source_id="synthetic-treehouse",
        source_cohort="SYNTHETIC_TREEHOUSE",
        cancer_code="CODE_A",
        tpm_file="treehouse.tsv",
        clinical_file="clinical.tsv",
        cohorts=(expression_builders.TreehouseCohort("CODE_A", "synthetic tumor"),),
    )

    def _fake_build(source_obj, *, cache_dir, output_dir=None, **_kwargs):
        assert source_obj is source
        assert Path(cache_dir) == tmp_path / "cache"
        assert output_dir is None
        matrix = pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["ENSG00000141510"],
                "Symbol": ["TP53"],
                "S1": [10.0],
            }
        )
        return expression_builders.SourceMatrixBuildResult(
            source=source,
            matrices={"CODE_A": matrix},
            matrix_paths={"CODE_A": tmp_path / "CODE_A_per_sample_tpm.parquet"},
            summary_rows=pd.DataFrame(
                columns=list(expression_builders.REFERENCE_EXPRESSION_COLUMNS)
            ),
            mapping_audit=pd.DataFrame(),
            parse_diagnostics=pd.DataFrame(),
            sample_qc=pd.DataFrame(),
            sidecar_paths={"mapping_audit": tmp_path / "mapping_audit.csv"},
        )

    monkeypatch.setattr(
        mod,
        "treehouse_source_from_registry",
        lambda source_id, registry_path=None: source,
    )
    monkeypatch.setattr(mod, "build_treehouse_source_matrices", _fake_build)

    assert mod.main(["synthetic-treehouse", "--cache-dir", str(tmp_path / "cache")]) == 0
    stdout = capsys.readouterr().out
    assert '"source_id": "synthetic-treehouse"' in stdout
    assert '"CODE_A": 1' in stdout


def test_build_gdc_script_uses_registry_config(tmp_path, monkeypatch, capsys):
    mod = _load_script("build_gdc_source")
    source = expression_builders.GdcSource(
        source_id="synthetic-gdc",
        project_ids=("TCGA-SYN",),
        source_cohort="SYNTHETIC_GDC",
        cancer_code="CODE_A",
    )

    def _fake_build(source_obj, *, cache_dir, output_dir=None, manifest=None, **_kwargs):
        assert source_obj is source
        assert Path(cache_dir) == tmp_path / "cache"
        assert output_dir is None
        assert manifest is None
        matrix = pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["ENSG00000141510"],
                "Symbol": ["TP53"],
                "S1": [10.0],
            }
        )
        return expression_builders.SourceMatrixBuildResult(
            source=source,
            matrices={"CODE_A": matrix},
            matrix_paths={"CODE_A": tmp_path / "CODE_A_per_sample_tpm.parquet"},
            summary_rows=pd.DataFrame(
                columns=list(expression_builders.REFERENCE_EXPRESSION_COLUMNS)
            ),
            mapping_audit=pd.DataFrame(),
            parse_diagnostics=pd.DataFrame(),
            sample_qc=pd.DataFrame(),
            sidecar_paths={"mapping_audit": tmp_path / "mapping_audit.csv"},
        )

    monkeypatch.setattr(
        mod,
        "gdc_source_from_registry",
        lambda source_id, registry_path=None: source,
    )
    monkeypatch.setattr(mod, "build_gdc_source_matrices", _fake_build)

    assert mod.main(["synthetic-gdc", "--cache-dir", str(tmp_path / "cache")]) == 0
    stdout = capsys.readouterr().out
    assert '"source_id": "synthetic-gdc"' in stdout
    assert '"CODE_A": 1' in stdout


def test_build_geo_matrix_script_derives_gene_lengths_for_raw_counts(
    tmp_path,
    monkeypatch,
):
    source_path = tmp_path / "source.tsv"
    pd.DataFrame(
        {
            "gene": ["ENSG00000141510", "ENSG00000146648"],
            "sample_1": [60, 80],
        }
    ).to_csv(source_path, sep="\t", index=False)
    registry_path = tmp_path / "expression_sources.yaml"
    registry_path.write_text(
        """
sources:
  - id: synthetic-counts
    source_type: geo-matrix
    cancer_codes: [CODE_A]
    file_url: https://example.org/source.tsv.gz
    file_name: source.tsv.gz
    unit: raw_counts
    gene_id_col: gene
    source_cohort: SYNTHETIC_COUNTS
""".lstrip()
    )
    mod = _load_script("build_geo_matrix")
    observed = {}

    def fake_gene_lengths(gene_ids, *, release):
        observed["gene_ids"] = list(gene_ids)
        observed["release"] = release
        return pd.Series(
            {
                "ENSG00000141510": 1.0,
                "ENSG00000146648": 2.0,
            }
        )

    monkeypatch.setattr(expression_builders, "ensembl_gene_lengths_kb", fake_gene_lengths)

    status = mod.main(
        [
            "--source-id",
            "synthetic-counts",
            "--registry",
            str(registry_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-path",
            str(source_path),
            "--ensembl-release",
            "112",
        ]
    )

    assert status == 0
    assert observed == {
        "gene_ids": ["ENSG00000141510", "ENSG00000146648"],
        "release": 112,
    }
    matrix = pd.read_parquet(tmp_path / "out" / "CODE_A_per_sample_tpm.parquet")
    by_gene = matrix.set_index("Ensembl_Gene_ID")
    assert by_gene.loc["ENSG00000141510", "sample_1"] == pytest.approx(600_000.0)
    assert by_gene.loc["ENSG00000146648", "sample_1"] == pytest.approx(400_000.0)


# ---------- cohort_percentile_vectors ----------


def test_percentile_vectors_schema_matches_reader():
    # 26 dense breakpoints p0..p100, plus the two id columns — the exact schema
    # expression.cohort_gene_percentiles reads back.
    genes = ["A", "B"]
    vals = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    out = expression_builders.cohort_percentile_vectors(_matrix(genes, list("wxyz"), vals))
    bp_cols = [c for c in out.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert len(bp_cols) == 26
    assert bp_cols[0] == "p0" and bp_cols[-1] == "p100"
    assert "p50" in bp_cols
    assert len(out) == 2


def test_percentile_vectors_log1p_roundtrip():
    # Stored log1p; expm1 of p0/p50/p100 recovers min/median/max of the gene's
    # across-sample distribution (the reader's as_tpm=True path).
    vals = np.array([[0.0, 10.0, 100.0, 1000.0]])
    out = expression_builders.cohort_percentile_vectors(_matrix(["A"], list("wxyz"), vals))
    row = out.iloc[0]
    assert np.isclose(np.expm1(np.float32(row["p0"])), 0.0, atol=1e-1)
    assert np.isclose(np.expm1(np.float32(row["p100"])), 1000.0, rtol=2e-2)
    # median of [0,10,100,1000] in log1p space, restored
    expected_med = np.median(np.log1p(vals[0]))
    assert np.isclose(np.float32(row["p50"]), expected_med, rtol=2e-2)


def test_percentile_vectors_ignores_nan():
    # A gene unmeasured in some samples: NaN cells are dropped, not treated as 0.
    vals = np.array([[np.nan, 10.0, 10.0, 10.0]])
    out = expression_builders.cohort_percentile_vectors(_matrix(["A"], list("wxyz"), vals))
    assert np.isclose(np.expm1(np.float32(out.iloc[0]["p50"])), 10.0, rtol=2e-2)


def test_percentile_vectors_requires_samples():
    with pytest.raises(ValueError):
        expression_builders.cohort_percentile_vectors(_matrix(["A"], [], np.empty((1, 0))))


# ---------- cohort_medoids ----------


def test_medoids_returns_base_plus_k():
    rng = np.arange(50.0).reshape(5, 10)  # 5 genes × 10 samples
    out = expression_builders.cohort_medoids(
        _matrix([f"g{i}" for i in range(5)], [f"s{j}" for j in range(10)], rng), k=3
    )
    rep_cols = [c for c in out.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert len(rep_cols) == 3
    assert list(out["Ensembl_Gene_ID"]) == [f"g{i}" for i in range(5)]


def test_medoids_small_cohort_keeps_all():
    vals = np.array([[1.0, 2.0]])
    out = expression_builders.cohort_medoids(_matrix(["A"], ["s1", "s2"], vals), k=5)
    assert [c for c in out.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == ["s1", "s2"]


def test_medoids_small_cohort_uses_stable_sample_id_order():
    vals = np.array([[2.0, 1.0]])
    out = expression_builders.cohort_medoids(_matrix(["A"], ["s2", "s1"], vals), k=5)
    assert [c for c in out.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == ["s1", "s2"]


def test_medoids_central_first_then_outlier():
    # 4 near-identical "typical" samples + 1 far outlier. The medoid (pick 1)
    # must come from the dense cluster; the farthest pick (2) must be the outlier.
    genes = [f"g{i}" for i in range(6)]
    typical = np.tile(np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]), (1, 4))
    typical = typical + np.array([[0, 0.01, -0.01, 0.0]] * 6)  # tiny jitter
    outlier = np.array([[100.0], [100.0], [100.0], [100.0], [100.0], [100.0]])
    vals = np.hstack([typical, outlier])
    samples = ["t1", "t2", "t3", "t4", "outlier"]
    out = expression_builders.cohort_medoids(_matrix(genes, samples, vals), k=2)
    picks = [c for c in out.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert picks[0] != "outlier"  # central medoid from the dense cluster
    assert picks[1] == "outlier"  # farthest-first grabs the outlier


def test_medoids_preserve_original_tpm():
    # Distance uses log1p internally, but stored values are the original TPM.
    vals = np.array([[7.0, 8.0, 9.0]])
    out = expression_builders.cohort_medoids(_matrix(["A"], ["s1", "s2", "s3"], vals), k=3)
    kept = out[[c for c in out.columns if c not in ("Ensembl_Gene_ID", "Symbol")]].to_numpy()
    assert set(kept.ravel()) == {7.0, 8.0, 9.0}


def test_medoids_can_select_on_biological_view_but_return_full_values():
    full = _matrix(
        ["BIO", "TECH"],
        ["s3", "s1", "s2"],
        np.array(
            [
                [10.0, 0.0, 5.0],
                [1000.0, 1000.0, 0.0],
            ]
        ),
    )
    biological = full[full["Ensembl_Gene_ID"] == "BIO"].reset_index(drop=True)

    out = expression_builders.cohort_medoids(
        full,
        sample_cols=["s3", "s1", "s2"],
        k=1,
        selection_df=biological,
    )

    rep_cols = [c for c in out.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert rep_cols == ["s2"]
    assert list(out["Ensembl_Gene_ID"]) == ["BIO", "TECH"]
    assert out.set_index("Ensembl_Gene_ID").loc["TECH", "s2"] == 0.0


def test_medoids_deterministic():
    rng = (np.arange(60.0) * 1.7 % 11).reshape(6, 10)
    df = _matrix([f"g{i}" for i in range(6)], [f"s{j}" for j in range(10)], rng)
    a = expression_builders.cohort_medoids(df, k=4)
    b = expression_builders.cohort_medoids(df, k=4)
    assert list(a.columns) == list(b.columns)


# ---------- generator scripts (end-to-end on a synthetic per-sample dir) ----------


def _write_cohort(tmp_path, code, genes, samples, values):
    df = _matrix(genes, samples, values)
    path = tmp_path / f"{code}.parquet"
    df.to_parquet(path, index=False)
    return path


def test_percentiles_generator_writes_readable_shards(tmp_path):
    gen = _load_script("generate_cohort_percentiles")
    inp = tmp_path / "in"
    inp.mkdir()
    _write_cohort(
        inp,
        "COHORT_A",
        ["A", "B"],
        list("wxyz"),
        np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]),
    )
    out = tmp_path / "out"
    gen.build(inp, drop_genes=set(), out_dir=out)
    shard = pd.read_parquet(out / "COHORT_A.parquet")
    assert len(shard) == 2
    bp_cols = [c for c in shard.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert len(bp_cols) == 26
    # p100 of gene A (max 4.0), stored log1p
    a = shard[shard["Ensembl_Gene_ID"] == "A"].iloc[0]
    assert np.isclose(np.expm1(np.float32(a["p100"])), 4.0, rtol=2e-2)


def test_percentiles_generator_drops_genes(tmp_path):
    gen = _load_script("generate_cohort_percentiles")
    inp = tmp_path / "in"
    inp.mkdir()
    _write_cohort(
        inp,
        "COHORT_A",
        ["KEEP", "DROPME"],
        ["s1", "s2"],
        np.array([[1.0, 2.0], [9.0, 9.0]]),
    )
    out = tmp_path / "out"
    gen.build(inp, drop_genes={"DROPME"}, out_dir=out)
    shard = pd.read_parquet(out / "COHORT_A.parquet")
    assert list(shard["Ensembl_Gene_ID"]) == ["KEEP"]


def test_representatives_generator_writes_shards_and_provenance(tmp_path):
    gen = _load_script("generate_representatives")
    inp = tmp_path / "in"
    inp.mkdir()
    # 6 samples so k=3 actually selects a subset
    vals = np.array([[float(g * 10 + s) for s in range(6)] for g in range(4)])
    _write_cohort(inp, "COHORT_A", [f"g{i}" for i in range(4)], [f"s{j}" for j in range(6)], vals)
    out = tmp_path / "out"
    gen.build(inp, k=3, out_dir=out)

    shard = pd.read_parquet(out / "COHORT_A.parquet")
    rep_cols = [c for c in shard.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert rep_cols == ["COHORT_A__rep1", "COHORT_A__rep2", "COHORT_A__rep3"]

    prov = pd.read_csv(out / "_provenance.csv")
    assert set(prov["representative_id"]) == set(rep_cols)
    assert (prov["n_cohort_samples"] == 6).all()
    # The reader merges on these exact columns — all must be present (source_project
    # is best-effort and empty for an unregistered synthetic code, but the column
    # must exist so consumers don't KeyError).
    for col in (
        "representative_id",
        "source_cohort",
        "source_project",
        "source_sample",
        "source_group_id",
        "n_cohort_samples",
    ):
        assert col in prov.columns
    # Unregistered code -> source_cohort falls back to the code itself.
    assert (prov["source_cohort"] == "COHORT_A").all()
    assert set(prov["source_sample"]) <= {f"s{i}" for i in range(6)}
    assert (prov["source_group_id"] == prov["source_cohort"] + ":" + prov["source_sample"]).all()
    assert prov["partition_role"].value_counts().to_dict() == {
        "train": 2,
        "validation": 1,
    }
    assert set(prov["partition_status"]) == {"available"}


def test_representatives_generator_selects_on_biological_view(tmp_path, monkeypatch):
    gen = _load_script("generate_representatives")
    inp = tmp_path / "in"
    inp.mkdir()
    _write_cohort(
        inp,
        "COHORT_A",
        ["BIO", "TECH"],
        ["sample_a", "sample_b"],
        np.array([[5.0, 10.0], [1000.0, 0.0]]),
    )
    monkeypatch.setattr(gen, "clean_tpm_censored_gene_ids", lambda: {"TECH"})
    seen = {}

    def fake_medoids(df, sample_cols, *, k, selection_df):
        seen["value_ids"] = df["Ensembl_Gene_ID"].tolist()
        seen["selection_ids"] = selection_df["Ensembl_Gene_ID"].tolist()
        seen["sample_cols"] = list(sample_cols)
        return df[["Ensembl_Gene_ID", "Symbol", sample_cols[0]]].copy()

    monkeypatch.setattr(gen, "cohort_medoids", fake_medoids)

    gen.build(inp, k=1, out_dir=tmp_path / "out")

    assert seen == {
        "value_ids": ["BIO", "TECH"],
        "selection_ids": ["BIO"],
        "sample_cols": ["sample_a", "sample_b"],
    }


def test_representatives_generator_groups_treehouse_views_by_physical_sample(tmp_path, monkeypatch):
    gen = _load_script("generate_representatives")
    inp = tmp_path / "in"
    inp.mkdir()
    _write_cohort(inp, "BRCA", ["BIO"], ["TCGA-AC-A2QH-01"], np.array([[5.0]]))
    monkeypatch.setattr(
        gen,
        "_cohort_provenance",
        lambda code: ("TREEHOUSE_POLYA_25_01_TCGA_SAMPLES", "Treehouse"),
    )

    out = tmp_path / "out"
    gen.build(inp, k=1, out_dir=out)

    provenance = pd.read_csv(out / "_provenance.csv").iloc[0]
    assert provenance["source_cohort"] == "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES"
    assert provenance["source_group_id"] == ("TREEHOUSE_POLYA_25_01:TCGA-AC-A2QH-01")


def test_partition_existing_bundle_updates_only_provenance_and_metadata(tmp_path):
    gen = _load_script("partition_representatives")
    representative_dir = tmp_path / "cancer-reference-expression-representatives"
    representative_dir.mkdir()
    shard_path = representative_dir / "A.parquet"
    shard_path.write_bytes(b"unchanged expression shard")
    pd.DataFrame(
        {
            "representative_id": [f"A__rep{i}" for i in range(1, 6)],
            "source_group_id": [f"PROJECT:s{i}" for i in range(1, 6)],
            "benchmark_eligible": [True] * 5,
        }
    ).to_csv(representative_dir / "_provenance.csv", index=False)
    pd.DataFrame(
        {
            "cancer_code": ["A"],
            "source_cohort": ["PROJECT"],
            "n_cohort_samples": [10],
        }
    ).to_csv(tmp_path / "expression-artifact-build-metadata.csv", index=False)
    (tmp_path / "expression-artifact-build-metadata.json").write_text(
        json.dumps(
            {
                "artifact": "expression-derived-shards",
                "schema_version": "expression_artifact_build_metadata_v2",
            }
        )
    )

    partitioned = gen.partition_existing_bundle(tmp_path)

    assert shard_path.read_bytes() == b"unchanged expression shard"
    assert partitioned["partition_role"].value_counts().to_dict() == {
        "train": 3,
        "validation": 2,
    }
    cohort_metadata = pd.read_csv(tmp_path / "expression-artifact-build-metadata.csv")
    assert cohort_metadata.loc[0, "partition_status"] == "available"
    assert cohort_metadata.loc[0, "n_partition_train"] == 3
    assert cohort_metadata.loc[0, "n_partition_validation"] == 2
    build_metadata = json.loads((tmp_path / "expression-artifact-build-metadata.json").read_text())
    assert build_metadata["schema_version"] == "expression_artifact_build_metadata_v3"
    assert build_metadata["representative_partition"]["role_counts"] == {
        "train": 3,
        "validation": 2,
    }


def _write_rebuild_inputs(tmp_path):
    cache = tmp_path / "cache"
    ref = tmp_path / "ref"
    source_dir = cache / "TEST_SOURCE" / "derived"
    source_dir.mkdir(parents=True)
    ref.mkdir()
    _matrix(
        ["ENSG000001", "ENSG000002", "ENSG000003"],
        ["pass_sample", "warn_sample", "fail_sample"],
        np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
                [70.0, 80.0, 90.0],
            ]
        ),
    ).to_parquet(source_dir / "X_per_sample_tpm.parquet", index=False)
    _write_cohort(ref, "X", ["ENSG000001"], ["reference"], np.array([[1.0]]))
    return cache, ref


def _patch_rebuild_registry(monkeypatch, gen):
    monkeypatch.setattr(
        gen,
        "source_registry",
        lambda: pd.DataFrame({"cancer_code": ["X"], "source_cohort": ["TEST_SOURCE"]}),
    )
    monkeypatch.setattr(gen, "cohort_source_version", lambda code: "test-source-version")
    monkeypatch.setattr(
        gen,
        "sample_expression_qc_from_matrix",
        lambda raw, cancer_type: pd.DataFrame(
            {
                "cancer_code": [cancer_type, cancer_type, cancer_type],
                "sample_id": ["pass_sample", "warn_sample", "fail_sample"],
                "sample_qc_status": ["pass", "warn", "fail"],
                "sample_qc_reasons": [
                    "",
                    "nonlinear_or_proxy_expression_scale",
                    "low_detected_genes",
                ],
                "source_scale_class": [
                    "linear_rnaseq_tpm",
                    "microarray_tpm_proxy",
                    "linear_rnaseq_tpm",
                ],
                "linear_tpm_comparable": [True, False, True],
                "recommended_for_absolute_tpm_floor": [True, False, False],
            }
        ),
    )


def test_rebuild_expression_artifacts_defaults_to_qc_passing_samples(tmp_path, monkeypatch):
    gen = _load_script("rebuild_expression_artifacts")
    cache, ref = _write_rebuild_inputs(tmp_path)
    _patch_rebuild_registry(monkeypatch, gen)
    out = tmp_path / "out"

    gen.rebuild(cache, ref, out, limit=None, validate=False, sample_qc="pass")

    clean = pd.read_parquet(out / "clean" / "X.parquet")
    assert [c for c in clean.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == ["pass_sample"]

    summary = pd.read_csv(out / "cancer-reference-expression" / "TEST_SOURCE.csv.gz")
    assert set(summary["n_samples"]) == {3}
    assert summary.set_index("Ensembl_Gene_ID").loc["ENSG000001", "TPM_median"] == 20.0

    reps = pd.read_parquet(out / "cancer-reference-expression-representatives" / "X.parquet")
    assert [c for c in reps.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == ["X__rep1"]

    prov = pd.read_csv(out / "cancer-reference-expression-representatives" / "_provenance.csv")
    assert prov.loc[0, "n_source_samples"] == 3
    assert prov.loc[0, "n_cohort_samples"] == 1
    assert prov.loc[0, "sample_qc"] == "pass"
    assert prov.loc[0, "sample_qc_requested"] == "pass"
    assert prov.loc[0, "source_sample_qc"] == "pass"
    assert prov.loc[0, "source_sample"] == "pass_sample"
    assert prov.loc[0, "source_group_id"] == "TEST_SOURCE:pass_sample"
    assert pd.isna(prov.loc[0, "source_diagnosis"])
    assert pd.isna(prov.loc[0, "source_morphology"])
    assert prov.loc[0, "representative_role"] == "standard"
    assert bool(prov.loc[0, "benchmark_eligible"]) is True
    assert pd.isna(prov.loc[0, "review_source"])
    assert pd.isna(prov.loc[0, "review_note"])
    assert prov.loc[0, "source_scale_class"] == "linear_rnaseq_tpm"
    assert bool(prov.loc[0, "linear_tpm_comparable"]) is True
    assert bool(prov.loc[0, "recommended_for_absolute_tpm_floor"]) is True
    assert prov.loc[0, "selection_scale_class"] == "linear_rnaseq_tpm"
    assert prov.loc[0, "sample_qc_policy_version"] == "sample_expression_qc_v2"
    assert prov.loc[0, "n_qc_pass"] == 1
    assert prov.loc[0, "n_qc_warn"] == 1
    assert prov.loc[0, "n_qc_fail"] == 1
    assert prov.loc[0, "partition_role"] == "train"
    assert prov.loc[0, "partition_status"] == "insufficient_independent_groups"
    assert prov.loc[0, "partition_validation_target"] == 0

    qc = pd.read_csv(out / "source-matrix-sample-qc.csv")
    assert list(qc["sample_id"]) == ["pass_sample", "warn_sample", "fail_sample"]

    build_meta = pd.read_csv(out / "expression-artifact-build-metadata.csv")
    assert build_meta.loc[0, "source_cohort"] == "TEST_SOURCE"
    assert build_meta.loc[0, "build_source_cohort"] == "TEST_SOURCE"
    assert build_meta.loc[0, "n_source_samples"] == 3
    assert build_meta.loc[0, "n_cohort_samples"] == 1
    assert build_meta.loc[0, "sample_qc_policy_version"] == "sample_expression_qc_v2"
    assert build_meta.loc[0, "partition_status"] == "insufficient_independent_groups"
    assert build_meta.loc[0, "n_partition_train"] == 1
    assert build_meta.loc[0, "n_partition_validation"] == 0

    metadata = json.loads((out / "expression-artifact-build-metadata.json").read_text())
    assert metadata["schema_version"] == "expression_artifact_build_metadata_v3"
    assert metadata["sample_qc"] == "pass"
    assert metadata["sample_qc_manifest"] == "source-matrix-sample-qc.csv"
    assert metadata["n_source_samples"] == 3
    assert metadata["n_cohort_samples"] == 1
    assert metadata["representative_partition"] == {
        "grouping_key": "source_group_id",
        "policy_version": "source_grouped_3_train_2_validation_v1",
        "role_counts": {"train": 1},
    }


def test_rebuild_expression_artifacts_applies_reviewed_source_adjudication(tmp_path, monkeypatch):
    gen = _load_script("rebuild_expression_artifacts")
    cache, ref = _write_rebuild_inputs(tmp_path)
    _patch_rebuild_registry(monkeypatch, gen)
    adjudication = gen._RepresentativeAdjudication(
        source_project="TCGA-BRCA",
        source_diagnosis="Metaplastic carcinoma, NOS",
        source_morphology="8575/3",
        representative_role="atypical_metaplastic_dual_lineage_audit_only",
        benchmark_eligible=False,
        review_source="https://example.test/review",
        review_note="reviewed physical source",
    )
    monkeypatch.setattr(
        gen,
        "_representative_adjudications",
        lambda: {"TEST_SOURCE:pass_sample": adjudication},
    )

    out = tmp_path / "out"
    gen.rebuild(cache, ref, out, limit=None, validate=False, sample_qc="pass")

    provenance = pd.read_csv(
        out / "cancer-reference-expression-representatives" / "_provenance.csv"
    ).iloc[0]
    assert provenance["source_project"] == "TCGA-BRCA"
    assert provenance["source_diagnosis"] == "Metaplastic carcinoma, NOS"
    assert provenance["source_morphology"] == "8575/3"
    assert provenance["representative_role"] == ("atypical_metaplastic_dual_lineage_audit_only")
    assert bool(provenance["benchmark_eligible"]) is False
    assert provenance["review_source"] == "https://example.test/review"
    assert provenance["review_note"] == "reviewed physical source"


def test_representative_source_adjudication_records_metaplastic_brca_source():
    gen = _load_script("rebuild_expression_artifacts")
    adjudication = gen._representative_adjudications()["TREEHOUSE_POLYA_25_01:TCGA-AC-A2QH-01"]

    assert adjudication.source_project == "TCGA-BRCA"
    assert adjudication.source_diagnosis == "Metaplastic carcinoma, NOS"
    assert adjudication.source_morphology == "8575/3"
    assert adjudication.representative_role == ("atypical_metaplastic_dual_lineage_audit_only")
    assert adjudication.benchmark_eligible is False
    assert "67c5f371-3fa9-47c5-8b15-c2dd9acc8519" in adjudication.review_source


def test_representative_source_adjudication_cannot_override_failed_qc():
    gen = _load_script("rebuild_expression_artifacts")
    adjudication = gen._RepresentativeAdjudication(
        source_project="TEST",
        source_diagnosis="Reviewed diagnosis",
        source_morphology="0000/0",
        representative_role="standard",
        benchmark_eligible=True,
        review_source="https://example.test/review",
        review_note="reviewed source",
    )

    with pytest.raises(ValueError, match="cannot override source QC"):
        gen._representative_provenance_fields(
            source_group_id="TEST_SOURCE:failed_sample",
            default_source_project="TEST",
            default_benchmark_eligible=False,
            adjudications={"TEST_SOURCE:failed_sample": adjudication},
        )


def test_rebuild_expression_artifacts_keeps_warn_proxy_source_when_pass_empty(
    tmp_path, monkeypatch
):
    gen = _load_script("rebuild_expression_artifacts")
    cache, ref = _write_rebuild_inputs(tmp_path)
    monkeypatch.setattr(
        gen,
        "source_registry",
        lambda: pd.DataFrame({"cancer_code": ["X"], "source_cohort": ["TEST_SOURCE"]}),
    )
    monkeypatch.setattr(gen, "cohort_source_version", lambda code: "test-source-version")
    monkeypatch.setattr(
        gen,
        "sample_expression_qc_from_matrix",
        lambda raw, cancer_type: pd.DataFrame(
            {
                "cancer_code": [cancer_type, cancer_type, cancer_type],
                "sample_id": ["pass_sample", "warn_sample", "fail_sample"],
                "sample_qc_status": ["warn", "warn", "warn"],
                "sample_qc_reasons": [
                    "nonlinear_or_proxy_expression_scale",
                    "nonlinear_or_proxy_expression_scale",
                    "nonlinear_or_proxy_expression_scale",
                ],
                "source_scale_class": [
                    "microarray_tpm_proxy",
                    "microarray_tpm_proxy",
                    "microarray_tpm_proxy",
                ],
                "linear_tpm_comparable": [False, False, False],
                "recommended_for_absolute_tpm_floor": [False, False, False],
            }
        ),
    )
    out = tmp_path / "out"

    gen.rebuild(cache, ref, out, limit=None, validate=False, sample_qc="pass")

    clean = pd.read_parquet(out / "clean" / "X.parquet")
    assert [c for c in clean.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == [
        "pass_sample",
        "warn_sample",
        "fail_sample",
    ]
    build_meta = pd.read_csv(out / "expression-artifact-build-metadata.csv")
    assert build_meta.loc[0, "sample_qc"] == "pass"
    assert build_meta.loc[0, "sample_qc_effective"] == "pass_or_warn"
    assert build_meta.loc[0, "sample_qc_fallback_reason"] == "no_pass_samples_tpm_proxy_source"
    prov = pd.read_csv(out / "cancer-reference-expression-representatives" / "_provenance.csv")
    assert set(prov["source_sample_qc_reasons"]) == {"nonlinear_or_proxy_expression_scale"}
    assert set(prov["source_scale_class"]) == {"microarray_tpm_proxy"}
    assert not prov["linear_tpm_comparable"].any()
    assert not prov["recommended_for_absolute_tpm_floor"].any()
    assert set(prov["selection_scale_class"]) == {"microarray_tpm_proxy"}
    metadata = json.loads((out / "expression-artifact-build-metadata.json").read_text())
    assert metadata["sample_qc_fallbacks"] == 1


def test_rebuild_expression_artifacts_keeps_concentration_only_source_when_pass_empty(
    tmp_path, monkeypatch
):
    gen = _load_script("rebuild_expression_artifacts")
    cache, ref = _write_rebuild_inputs(tmp_path)
    monkeypatch.setattr(
        gen,
        "source_registry",
        lambda: pd.DataFrame({"cancer_code": ["X"], "source_cohort": ["TEST_SOURCE"]}),
    )
    monkeypatch.setattr(gen, "cohort_source_version", lambda code: "test-source-version")
    monkeypatch.setattr(
        gen,
        "sample_expression_qc_from_matrix",
        lambda raw, cancer_type: pd.DataFrame(
            {
                "cancer_code": [cancer_type, cancer_type, cancer_type],
                "sample_id": ["pass_sample", "warn_sample", "fail_sample"],
                "sample_qc_status": ["fail", "fail", "fail"],
                "sample_qc_reasons": [
                    "high_top10_gene_fraction",
                    "high_top_gene_fraction;high_top10_gene_fraction",
                    "high_top10_gene_fraction",
                ],
            }
        ),
    )
    out = tmp_path / "out"

    gen.rebuild(cache, ref, out, limit=None, validate=False, sample_qc="pass")

    clean = pd.read_parquet(out / "clean" / "X.parquet")
    assert [c for c in clean.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == [
        "pass_sample",
        "warn_sample",
        "fail_sample",
    ]
    build_meta = pd.read_csv(out / "expression-artifact-build-metadata.csv")
    assert build_meta.loc[0, "sample_qc"] == "pass"
    assert build_meta.loc[0, "sample_qc_effective"] == "all"
    assert (
        build_meta.loc[0, "sample_qc_fallback_reason"]
        == "no_pass_samples_high_concentration_source"
    )
    prov = pd.read_csv(out / "cancer-reference-expression-representatives" / "_provenance.csv")
    assert set(prov["sample_qc"]) == {"fail"}
    assert set(prov["sample_qc_requested"]) == {"pass"}
    assert set(prov["source_sample_qc"]) == {"fail"}
    assert set(prov["representative_role"]) == {"source_qc_fallback_audit_only"}
    assert not prov["benchmark_eligible"].any()
    assert prov["source_sample"].notna().all()
    assert prov["source_group_id"].str.startswith("TEST_SOURCE:").all()


def test_rebuild_expression_artifacts_clips_negative_source_values():
    gen = _load_script("rebuild_expression_artifacts")
    df = _matrix(["G1", "G2"], ["s1", "s2"], np.array([[-2.0, 3.0], [4.0, -0.5]]))

    out, n_negative = gen._clip_negative_expression(df, ["s1", "s2"])

    assert n_negative == 2
    assert out[["s1", "s2"]].to_numpy().min() == 0.0
    assert df[["s1", "s2"]].to_numpy().min() < 0.0


def test_rebuild_expression_artifacts_disambiguates_source_by_registry_sample_count(
    tmp_path, monkeypatch
):
    gen = _load_script("rebuild_expression_artifacts")
    cache = tmp_path / "cache"
    ref = tmp_path / "ref"
    ref.mkdir()
    _write_cohort(ref, "X", ["G1", "G2"], ["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    source_a = cache / "source-a" / "derived"
    source_b = cache / "source-b" / "derived"
    source_a.mkdir(parents=True)
    source_b.mkdir(parents=True)
    _matrix(["G1", "G2"], ["s1"], np.array([[1.0], [2.0]])).to_parquet(
        source_a / "X_per_sample_tpm.parquet", index=False
    )
    _matrix(["G1", "G2"], ["s1", "s2"], np.array([[1.0, 2.0], [2.0, 3.0]])).to_parquet(
        source_b / "X_per_sample_tpm.parquet", index=False
    )
    monkeypatch.setattr(
        gen,
        "source_registry",
        lambda: pd.DataFrame(
            {"cancer_code": ["X"], "source_cohort": ["AGGREGATE_LABEL"], "n_samples": [2]}
        ),
    )
    monkeypatch.setattr(gen, "cohort_source_version", lambda code: "test-source-version")
    monkeypatch.setattr(
        gen,
        "sample_expression_qc_from_matrix",
        lambda raw, cancer_type: pd.DataFrame(
            {
                "cancer_code": [cancer_type] * (len(raw.columns) - 2),
                "sample_id": [c for c in raw.columns if c not in ("Ensembl_Gene_ID", "Symbol")],
                "sample_qc_status": ["pass"] * (len(raw.columns) - 2),
                "sample_qc_reasons": [""] * (len(raw.columns) - 2),
            }
        ),
    )

    gen.rebuild(cache, ref, tmp_path / "out", limit=None, validate=False, sample_qc="pass")

    build_meta = pd.read_csv(tmp_path / "out" / "expression-artifact-build-metadata.csv")
    assert build_meta.loc[0, "n_source_samples"] == 2
    assert build_meta.loc[0, "source_matrix_path"].endswith(
        "source-b/derived/X_per_sample_tpm.parquet"
    )


def test_rebuild_expression_artifacts_prefers_aggregate_source_directory(tmp_path):
    gen = _load_script("rebuild_expression_artifacts")
    preferred = tmp_path / "geo-heme" / "derived" / "X_per_sample_tpm.parquet"
    duplicate = tmp_path / "gse100026-cml" / "derived" / "X_per_sample_tpm.parquet"
    preferred.parent.mkdir(parents=True)
    duplicate.parent.mkdir(parents=True)
    _matrix(["G1"], ["s1"], np.array([[1.0]])).to_parquet(preferred, index=False)
    _matrix(["G1"], ["s1"], np.array([[1.0]])).to_parquet(duplicate, index=False)

    selected = gen._select_source(
        "X",
        [("gse100026-cml", duplicate), ("geo-heme", preferred)],
        {"X": "GEO_HEME_2022"},
        {"X": 1},
    )

    assert selected == preferred


def test_rebuild_expression_artifacts_selects_representatives_on_biological_view(
    tmp_path, monkeypatch
):
    gen = _load_script("rebuild_expression_artifacts")
    cache = tmp_path / "cache"
    ref = tmp_path / "ref"
    source_dir = cache / "TEST_SOURCE" / "derived"
    source_dir.mkdir(parents=True)
    ref.mkdir()
    _matrix(
        ["BIO", "TECH"],
        ["sample_a", "sample_b"],
        np.array([[5.0, 10.0], [1000.0, 0.0]]),
    ).to_parquet(source_dir / "X_per_sample_tpm.parquet", index=False)
    _write_cohort(ref, "X", ["BIO"], ["reference"], np.array([[1.0]]))

    monkeypatch.setattr(
        gen,
        "source_registry",
        lambda: pd.DataFrame({"cancer_code": ["X"], "source_cohort": ["TEST_SOURCE"]}),
    )
    monkeypatch.setattr(gen, "cohort_source_version", lambda code: "test-source-version")
    monkeypatch.setattr(
        gen,
        "sample_expression_qc_from_matrix",
        lambda raw, cancer_type: pd.DataFrame(
            {
                "cancer_code": [cancer_type, cancer_type],
                "sample_id": ["sample_a", "sample_b"],
                "sample_qc_status": ["pass", "pass"],
                "sample_qc_reasons": ["", ""],
            }
        ),
    )
    monkeypatch.setattr(gen, "clean_tpm", lambda values, gene_table: values.copy())
    monkeypatch.setattr(gen, "clean_tpm_censored_gene_ids", lambda: {"TECH"})

    seen = {}

    def fake_medoids(df, sample_cols, *, k, selection_df):
        seen["value_ids"] = df["Ensembl_Gene_ID"].tolist()
        seen["selection_ids"] = selection_df["Ensembl_Gene_ID"].tolist()
        seen["sample_cols"] = list(sample_cols)
        return df[["Ensembl_Gene_ID", "Symbol", sample_cols[0]]].copy()

    monkeypatch.setattr(gen, "cohort_medoids", fake_medoids)

    gen.rebuild(cache, ref, tmp_path / "out", limit=None, validate=False, sample_qc="pass")

    assert seen == {
        "value_ids": ["BIO", "TECH"],
        "selection_ids": ["BIO"],
        "sample_cols": ["sample_a", "sample_b"],
    }


def test_rebuild_expression_artifacts_keeps_all_samples_when_requested(tmp_path, monkeypatch):
    gen = _load_script("rebuild_expression_artifacts")
    cache, ref = _write_rebuild_inputs(tmp_path)
    _patch_rebuild_registry(monkeypatch, gen)
    out = tmp_path / "out"

    gen.rebuild(cache, ref, out, limit=None, validate=False, sample_qc="all")

    clean = pd.read_parquet(out / "clean" / "X.parquet")
    assert [c for c in clean.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == [
        "pass_sample",
        "warn_sample",
        "fail_sample",
    ]
    build_meta = pd.read_csv(out / "expression-artifact-build-metadata.csv")
    assert build_meta.loc[0, "sample_qc"] == "all"
    assert build_meta.loc[0, "n_source_samples"] == 3
    assert build_meta.loc[0, "n_cohort_samples"] == 3


# ---------- real-data parity (skipped without the maintainer's matrix cache) ----------


@pytest.mark.skipif(not _PARITY_READY, reason="per-sample matrix cache / pirlygenes ref absent")
def test_percentiles_reproduce_pirlygenes_reference():
    # End-to-end on REAL data: raw per-sample matrix -> clean_tpm -> percentile
    # vectors must reproduce pirlygenes' shipped percentile artifact for the same
    # cohort. Proves the generator + oncoref's clean_tpm port are faithful.
    from oncoref import normalization as nz

    raw = pd.read_parquet(_ACC_MATRIX[0])
    samples = [c for c in raw.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    gene_table = raw[["Symbol", "Ensembl_Gene_ID"]]
    clean = nz.clean_tpm(raw[samples], gene_table=gene_table)
    clean_df = pd.concat([gene_table, clean], axis=1)

    mine = expression_builders.cohort_percentile_vectors(clean_df, samples).set_index(
        "Ensembl_Gene_ID"
    )
    ref = pd.read_parquet(_ACC_REF).set_index("Ensembl_Gene_ID")
    # Column schema is identical.
    assert [c for c in mine.columns if c != "Symbol"] == [c for c in ref.columns if c != "Symbol"]

    common = mine.index.intersection(ref.index)
    assert len(common) > 10_000
    # The deterministic mid/upper percentiles match (expm1 back to TPM); tiny tail
    # deviation at p99 is float16 rounding, so correlation must be essentially 1.
    for col in ("p50", "p95"):
        a = np.expm1(mine.loc[common, col].astype("float32"))
        b = np.expm1(ref.loc[common, col].astype("float32"))
        mask = (a > 0) | (b > 0)
        assert np.corrcoef(a[mask], b[mask])[0, 1] > 0.999
