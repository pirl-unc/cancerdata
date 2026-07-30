# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path

import pandas as pd

import oncoref
from oncoref import expression_builders, samples
from oncoref import expression_registry as es
from oncoref.load_dataset import get_data


def test_registry_loads_all_sources():
    srcs = es.expression_sources()
    assert len(srcs) > 50
    # every source has an id, a type, and at least one cancer code
    assert all(s.id and s.source_type and s.cancer_codes for s in srcs)


def test_source_types_cover_the_major_providers():
    types = {s.source_type for s in es.expression_sources()}
    for t in ("gdc", "treehouse-compendium", "recount3", "sra-salmon", "broad-cllmap"):
        assert t in types


def test_lookup_by_id_and_code():
    src = es.expression_source("mmrf-commpass")
    assert src is not None
    assert "MM" in src.cancer_codes
    assert src.source_type == "gdc"
    assert es.expression_source("not-a-source") is None
    assert any(s.id == "mmrf-commpass" for s in es.sources_for_cancer_code("MM"))


def test_ess_artifact_source_has_typed_provenance():
    source = es.expression_source("gse85383-ess")

    assert source is not None
    assert source.cancer_codes == ("SARC_ESS_LG", "SARC_ESS_HG")
    assert source.source_cohort == "GSE85383_YOSHIDA_2017_ESS"
    assert source.source_project == "GEO"
    assert source.source_type == "geo-microarray"
    assert source.unit == "TPM proxy"
    assert source.tumor_origin == "primary"
    assert source.processing_pipeline


def test_mbl_subgroup_source_has_typed_derivation_provenance():
    source = es.expression_source("treehouse-polya-25-01-mbl-subgroup-markers")

    assert source is not None
    assert source.cancer_codes == ("MBL_WNT", "MBL_SHH", "MBL_G3", "MBL_G4")
    assert source.source_cohort == "TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS"
    assert source.source_project == "Treehouse"
    assert source.source_type == "treehouse-derived"
    assert source.source_version == "25.01"
    assert source.unit == "TPM"
    assert source.tumor_origin == "primary"
    assert source.processing_pipeline


def test_mmnst_source_has_typed_raw_read_provenance():
    source = es.expression_source("prjna1083972-mmnst")

    assert source is not None
    assert source.cancer_codes == ("SARC_MMNST",)
    assert source.source_type == "sra-salmon"
    assert source.accession == "PRJNA1083972"
    assert source.source_cohort == "SRP493407_MMNST_2024"
    assert source.source_project == "NCBI SRA / ENA"
    assert source.unit == "Salmon gene TPM"
    assert source.tumor_origin == "primary"
    assert source.processing_pipeline


def test_gse294016_source_uses_authoritative_histology_mapping():
    entries = es.expression_source_registry_entries()
    source = next(row for row in entries if row["id"] == "gse294016-salivary-histology")
    typed_source = es.expression_source(source["id"])
    build_source = expression_builders.geo_matrix_source_from_registry(source["id"])
    mapping_path = (
        Path(__file__).resolve().parents[1]
        / "oncoref"
        / "data"
        / source["sample_to_cancer_code"]["mapping_file"]
    )
    mapping = pd.read_csv(mapping_path, keep_default_na=False)

    assert source["cancer_codes"] == ["ADCC", "ACINIC"]
    assert source["expected_source_samples"] == 95
    assert source["expected_samples_by_code"] == {"ADCC": 57, "ACINIC": 3}
    assert typed_source is not None
    assert typed_source.source_version == (
        "Bartl 2025 Supplementary Dataset 1 Table 1 diagnosis mapping"
    )
    assert typed_source.tumor_origin == "mixed"
    assert typed_source.processing_pipeline
    assert build_source.expected_source_samples == 95
    assert build_source.expected_samples_by_code == {"ADCC": 57, "ACINIC": 3}
    assert build_source.sample_to_cancer_code is not None
    assert build_source.sample_to_cancer_code("P-58.1") == "ADCC"
    assert build_source.sample_to_cancer_code("P-76") == "ACINIC"
    assert build_source.sample_to_cancer_code("P-89") is None
    assert len(mapping) == 95
    assert mapping["sample_id"].is_unique
    assert mapping["source_sample_id"].nunique() == 93
    assert mapping["cancer_code"].value_counts().to_dict() == {
        "ADCC": 57,
        "": 35,
        "ACINIC": 3,
    }
    by_sample = mapping.set_index("sample_id")
    assert by_sample.loc["P-58.1", "source_sample_id"] == "P-58"
    assert by_sample.loc["P-77.1", "source_sample_id"] == "P-77"
    expected_route = mapping["cancer_code"].replace("", None).tolist()
    actual_route = mapping["sample_id"].map(build_source.sample_to_cancer_code).tolist()
    assert actual_route == expected_route

    expected_counts = source["expected_samples_by_code"]
    source_counts = (
        get_data("source-matrices")
        .set_index("cancer_code")
        .loc[list(expected_counts), "n_samples"]
        .to_dict()
    )
    availability_counts = (
        get_data("cancer-reference-expression-availability")
        .set_index("cancer_code")
        .loc[list(expected_counts), "n_reference_samples"]
        .to_dict()
    )
    cohort_row = oncoref.cohort_registry_df().set_index("cohort_id").loc[source["source_cohort"]]
    public_availability = oncoref.cancer_reference_expression_availability(
        list(expected_counts),
        reference_source="summary_rows_all",
        sample_qc="all",
        all_sources=True,
    )
    public_counts = (
        public_availability.loc[public_availability["source_cohort"].eq(source["source_cohort"])]
        .set_index("cancer_code")["n_reference_samples"]
        .astype(int)
        .to_dict()
    )
    assert source_counts == expected_counts
    assert availability_counts == expected_counts
    assert public_counts == expected_counts
    assert int(cohort_row["n_samples"]) == source["expected_source_samples"]
    assert int(cohort_row["n_codes"]) == len(source["cancer_codes"])


def test_expression_sources_df_shape():
    df = es.expression_sources_df()
    assert {
        "id",
        "source_type",
        "cancer_codes",
        "source_cohort",
        "source_project",
        "processing_pipeline",
        "citation",
    } <= set(df.columns)
    assert len(df) == len(es.expression_sources())


def test_expression_source_registry_raw_helpers_are_public():
    path = es.expression_source_registry_path()
    assert path.name == "expression_sources.yaml"
    assert path.exists()

    text = es.expression_source_registry_text()
    assert "sources:" in text
    assert "source_type: geo-matrix" in text

    entries = es.expression_source_registry_entries()
    assert len(entries) == len(es.expression_sources())
    assert any(entry["id"] == "gse328026-sarc-pec" for entry in entries)

    geo = es.expression_source_registry_entries(source_type="geo-matrix")
    assert geo
    assert {entry["source_type"] for entry in geo} == {"geo-matrix"}
    assert geo == tuple(expression_builders.geo_matrix_source_entries())
    assert es.expression_source_registry_entries(source_type=["not-a-source-type"]) == ()
    assert oncoref.expression_source_registry_entries() == entries
    assert "expression_source_registry_entries" in oncoref.__all__


def test_sample_manifest_loads():
    df = samples.sample_manifest()
    assert len(df) > 1000
    for col in ("cancer_code", "source_cohort", "sample_id", "included"):
        assert col in df.columns


def test_samples_for_cancer_code_included_only():
    inc = samples.samples_for_cancer_code("BL", included_only=True)
    allrows = samples.samples_for_cancer_code("BL", included_only=False)
    assert len(inc) <= len(allrows)
    assert (inc["included"].astype(str).str.lower() == "true").all()


def test_sample_counts_sum():
    counts = samples.sample_counts_by_cancer_code()
    assert counts.sum() > 1000
    assert (counts > 0).all()
