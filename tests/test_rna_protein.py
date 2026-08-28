# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

import oncoref
from oncoref import rna_protein

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_rna_protein_calibration.py"
_SPEC = importlib.util.spec_from_file_location("build_rna_protein_calibration", _SCRIPT_PATH)
builder = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = builder
_SPEC.loader.exec_module(builder)


def test_cptac_source_manifest_is_complete_and_pinned():
    frame = rna_protein.rna_protein_calibration_sources()

    assert tuple(frame.columns) == rna_protein.RNA_PROTEIN_SOURCE_COLUMNS
    assert len(frame) == 20
    assert set(frame["cptac_cohort"]) == set(rna_protein.CPTAC_CALIBRATION_COHORTS)
    modalities = frame.groupby("cptac_cohort")["modality"].apply(set)
    assert all(value == {"rna", "protein"} for value in modalities)
    assert set(frame["source_record"]) == {"10.5281/zenodo.8394329"}
    assert set(frame["license_id"]) == {"CC-BY-4.0"}
    assert set(frame["processing_version"]) == {"1.5.14"}
    assert set(frame["processing_commit"]) == {"2255e5d20cf66ac8072ccfbb66519d4f7e7fb06a"}


def test_cptac_source_manifest_filters_are_exact_and_case_insensitive():
    ucec_rna = rna_protein.rna_protein_calibration_sources(cptac_cohort=" ucec ", modality="RNA")

    assert ucec_rna["source_id"].tolist() == ["CPTAC_BCM_UCEC_RNA"]
    assert rna_protein.rna_protein_calibration_sources(modality="ihc").empty


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda frame: frame.__setitem__("source_id", ["same"] * len(frame)), "source_id"),
        (
            lambda frame: frame.__setitem__(
                "source_scale", ["wrong", *frame["source_scale"].iloc[1:].tolist()]
            ),
            "source_scale",
        ),
        (
            lambda frame: frame.__setitem__(
                "checksum", ["sha256:no", *frame["checksum"].iloc[1:].tolist()]
            ),
            "checksums",
        ),
    ],
)
def test_source_manifest_validation_rejects_integrity_breaks(mutation, match):
    frame = rna_protein.rna_protein_calibration_sources()
    mutation(frame)

    with pytest.raises(ValueError, match=match):
        rna_protein._validate_rna_protein_calibration_sources(frame)


def test_source_manifest_validation_requires_both_modalities():
    frame = rna_protein.rna_protein_calibration_sources()
    frame = frame.loc[frame["source_id"].ne("CPTAC_BCM_UCEC_PROTEIN")]

    with pytest.raises(ValueError, match="one RNA and one protein"):
        rna_protein._validate_rna_protein_calibration_sources(frame)


def test_source_file_verification_checks_size_and_digest(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"matched-source")
    source = pd.Series(
        {
            "source_id": "synthetic",
            "byte_size": path.stat().st_size,
            "checksum": "md5:" + hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest(),
        }
    )

    builder.verify_source_file(path, source)
    source["checksum"] = "md5:" + "0" * 32
    with pytest.raises(ValueError, match="MD5 mismatch"):
        builder.verify_source_file(path, source)


def test_matched_cptac_join_is_canonical_deterministic_and_lossless_for_missingness(
    tmp_path,
):
    tp53 = "ENSG00000141510.17"
    egfr = "ENSG00000146648.16"
    retired_ggnbp2 = "ENSG00000005955.7"
    current_ggnbp2 = "ENSG00000278311.4"
    multi_gene = f"{tp53};{egfr}"
    unknown = "ENSG99999999999.1"
    genes = [tp53, egfr, retired_ggnbp2, current_ggnbp2, multi_gene, unknown]

    rna = pd.DataFrame(
        {
            "S2_T": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "S1": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
            "S3_A": [20.0] * 6,
            "RNA_ONLY_T": [30.0] * 6,
        },
        index=genes,
    )
    protein = pd.DataFrame(
        {
            "S1": [10.0, float("nan"), 50.0, 70.0, 90.0, 110.0],
            "S2_T": [20.0, 40.0, 60.0, 80.0, 100.0, 120.0],
            "PROTEIN_ONLY": [1.0] * 6,
        },
        index=genes,
    )
    rna_path = tmp_path / "rna.tsv"
    protein_path = tmp_path / "protein.tsv"
    rna.to_csv(rna_path, sep="\t")
    protein.to_csv(protein_path, sep="\t")

    first = builder.load_matched_cptac_cohort(rna_path, protein_path)
    second = builder.load_matched_cptac_cohort(rna_path, protein_path)

    pd.testing.assert_frame_equal(first.rna, second.rna)
    pd.testing.assert_frame_equal(first.protein, second.protein)
    assert first.samples["sample_id"].tolist() == ["S1", "S2"]
    assert first.samples["rna_source_sample_id"].tolist() == ["S1", "S2_T"]
    assert first.samples["protein_source_sample_id"].tolist() == ["S1", "S2_T"]
    assert first.genes.to_dict(orient="records") == [
        {"canonical_gene_id": "ENSG00000141510", "source_gene_id": tp53},
        {"canonical_gene_id": "ENSG00000146648", "source_gene_id": egfr},
    ]
    assert pd.isna(first.protein.loc["ENSG00000146648", "S1"])
    assert first.rna.loc["ENSG00000141510", "S1"] == 1.0
    reasons = first.excluded_genes.set_index("source_gene_id")["exclusion_reason"].to_dict()
    assert reasons[multi_gene] == "multi_gene_group"
    assert reasons[unknown] == "unmapped_gene_id"
    assert reasons[retired_ggnbp2] == "duplicate_canonical_gene"
    assert reasons[current_ggnbp2] == "duplicate_canonical_gene"


def test_rna_protein_source_api_is_exported():
    assert oncoref.rna_protein is rna_protein
    assert oncoref.rna_protein_calibration_sources is (rna_protein.rna_protein_calibration_sources)
