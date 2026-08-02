import pandas as pd
import pytest
from scripts.generate_reference_availability import build_reference_availability


def _write_shard(path, code, source, genes, n_samples):
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": genes,
            "cancer_code": code,
            "source_cohort": source,
            "source_project": "PROJECT",
            "source_version": "v1",
            "tumor_origin": "primary",
            "metastasis_site": pd.NA,
            "processing_pipeline": "rnaseq",
            "notes": "line one\nline two",
            "n_samples": n_samples,
        }
    ).to_csv(path, index=False)


def test_build_reference_availability_uses_declared_source_not_richest(tmp_path):
    _write_shard(tmp_path / "small.csv", "X", "SMALL", ["E1", "E2"], 20)
    _write_shard(tmp_path / "rich.csv", "X", "RICH", ["E1", "E2", "E3"], 5)
    _write_shard(tmp_path / "other.csv", "Y", "OTHER", ["E1"], 4)

    table = build_reference_availability(
        tmp_path,
        chunksize=1,
        selected_sources={"X": "SMALL", "Y": "OTHER"},
    )

    x = table[table["cancer_code"] == "X"].set_index("source_cohort")
    assert x.loc["RICH", "n_reference_genes"] == 3
    assert not x.loc["RICH", "selected"]
    assert x.loc["SMALL", "selected"]
    assert x.loc["SMALL", "n_reference_samples"] == 20
    assert x.loc["SMALL", "notes"] == "line one line two"


def test_build_reference_availability_rejects_split_source_identity(tmp_path):
    _write_shard(tmp_path / "one.csv", "X", "SAME", ["E1"], 1)
    _write_shard(tmp_path / "two.csv", "X", "SAME", ["E2"], 1)

    with pytest.raises(ValueError, match="split across multiple shards"):
        build_reference_availability(
            tmp_path,
            chunksize=1,
            selected_sources={"X": "SAME"},
        )


def test_build_reference_availability_rejects_missing_declared_source(tmp_path):
    _write_shard(tmp_path / "other.csv", "X", "OTHER", ["E1"], 1)

    with pytest.raises(ValueError, match="selected source rows are absent"):
        build_reference_availability(
            tmp_path,
            chunksize=1,
            selected_sources={"X": "SELECTED"},
        )


def test_build_reference_availability_rejects_missing_declared_code(tmp_path):
    _write_shard(tmp_path / "present.csv", "X", "PRESENT", ["E1"], 1)

    with pytest.raises(ValueError, match=r"\('Y', 'MISSING'\)"):
        build_reference_availability(
            tmp_path,
            chunksize=1,
            selected_sources={"X": "PRESENT", "Y": "MISSING"},
        )


def test_build_reference_availability_routes_all_sarc_histology_overlays(tmp_path):
    legacy = "TREEHOUSE_POLYA_25_01_TCGA_SUBSET"
    generic = "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES"
    histology = "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY"
    _write_shard(tmp_path / "ddlps.csv", "SARC_DDLPS", legacy, ["E1"], 48)
    _write_shard(tmp_path / "wdlps.csv", "SARC_WDLPS", legacy, ["E1"], 5)
    _write_shard(tmp_path / "pleolps.csv", "SARC_PLEOLPS", generic, ["E1"], 2)
    _write_shard(tmp_path / "luad.csv", "LUAD", legacy, ["E1"], 20)

    table = build_reference_availability(
        tmp_path,
        chunksize=1,
        selected_sources={
            "SARC_DDLPS": histology,
            "SARC_PLEOLPS": histology,
            "SARC_WDLPS": histology,
            "LUAD": generic,
        },
    )
    by_code = table.set_index("cancer_code")["source_cohort"]

    assert by_code.loc[["SARC_DDLPS", "SARC_PLEOLPS", "SARC_WDLPS"]].eq(histology).all()
    assert by_code.loc["LUAD"] == generic
