# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Source-anchored replacement and migration audit for legacy driver tables."""

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

import oncoref
from oncoref import drivers

_ROOT = Path(__file__).resolve().parents[1]


def test_driver_evidence_is_a_complete_source_identical_migration():
    genes = drivers.driver_gene_evidence_df()
    variants = drivers.driver_variant_evidence_df()

    assert len(genes) == 739
    assert len(variants) == 579
    assert genes["record_id"].is_unique
    assert variants["record_id"].is_unique
    assert set(genes["scope_kind"]) == {"pan_cancer", "source_cancer_code"}
    assert set(variants["scope_kind"]) == {"pan_cancer"}
    assert set(genes["source_locator"]) == {"Table S1"}
    assert set(variants["source_locator"]) == {"Table S4"}
    assert set(genes["pmid"]) == {"PMID:29625053"}
    assert set(variants["pmid"]) == {"PMID:29625053"}

    audit = drivers.driver_legacy_migration_audit_df()
    assert len(audit) == 739 + 579
    assert set(audit["migration_status"]) == {"migrated"}
    assert audit["legacy_key"].str.strip().ne("").all()
    assert audit["destination_record_id"].str.strip().ne("").all()
    assert audit["source_locator"].isin({"Table S1", "Table S4"}).all()

    summary = drivers.driver_legacy_migration_summary().set_index("legacy_dataset")
    assert summary.loc["cancer-driver-genes"].to_dict() == {
        "total": 739,
        "awaiting_source": 0,
        "migrated": 739,
        "rejected": 0,
    }
    assert summary.loc["cancer-driver-variants"].to_dict() == {
        "total": 579,
        "awaiting_source": 0,
        "migrated": 579,
        "rejected": 0,
    }


def test_driver_evidence_preserves_gene_transcript_and_variant_identity():
    tp53 = drivers.driver_variant_evidence("TP53", protein_change="p.R175H")
    assert len(tp53) == 1
    assert tp53.iloc[0]["ensembl_gene_id"] == "ENSG00000141510"
    assert tp53.iloc[0]["ensembl_transcript_id"] == "ENST00000269305"
    assert tp53.iloc[0]["variant_notation"] == "HGVS protein"
    assert tp53.iloc[0]["scope_kind"] == "pan_cancer"

    kras = drivers.driver_variant_evidence("KRAS", protein_change="p.G12V")
    assert len(kras) == 1
    assert kras.iloc[0]["ensembl_gene_id"] == "ENSG00000133703"
    assert kras.iloc[0]["source_scope"] == "TCGA PanCancer Atlas"

    kras_luad = drivers.driver_gene_evidence("KRAS", source_scope="luad")
    assert len(kras_luad) == 1
    assert kras_luad.iloc[0]["source_scope"] == "LUAD"
    assert kras_luad.iloc[0]["scope_kind"] == "source_cancer_code"


def test_driver_evidence_accessors_are_public_and_defensive():
    assert oncoref.driver_gene_evidence_df is drivers.driver_gene_evidence_df
    assert oncoref.driver_variant_evidence_df is drivers.driver_variant_evidence_df
    first = oncoref.driver_gene_evidence_df()
    first.loc[0, "gene_symbol"] = "changed"
    assert oncoref.driver_gene_evidence_df().loc[0, "gene_symbol"] == "ABL1"


def test_driver_evidence_builder_pins_inputs_and_generated_outputs():
    result = subprocess.run(
        [sys.executable, "scripts/build_driver_evidence.py", "--check"],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    expected = {
        "cancer-driver-genes.csv": (
            "dbfb76d906e218dab2d94e90dd51f8db41dbccf6c82da11311baa418ea738fdf"
        ),
        "cancer-driver-variants.csv": (
            "d568385b221f0e28f619e3358b42cec7449b6d434237516df871b7f8e9ed4549"
        ),
    }
    for name, digest in expected.items():
        observed = hashlib.sha256((_ROOT / "oncoref" / "data" / name).read_bytes()).hexdigest()
        assert observed == digest


def test_driver_migration_audit_rejects_unknown_dataset():
    try:
        drivers.driver_legacy_migration_audit_df("not-a-driver-table")
    except ValueError as error:
        assert "dataset must be one of" in str(error)
    else:
        raise AssertionError("unknown driver table should fail")


def test_driver_evidence_filters_return_empty_frames_for_unknown_genes():
    assert drivers.driver_gene_evidence("NOT_A_REAL_GENE").empty
    assert drivers.driver_variant_evidence("NOT_A_REAL_GENE").empty
    assert isinstance(drivers.driver_gene_evidence(), pd.DataFrame)
