# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import re
from pathlib import Path

import pandas as pd

import oncoref
from oncoref import expression_builders, samples, source_matrices
from oncoref import expression_registry as es
from oncoref.cancer_types import resolve_cancer_type
from oncoref.load_dataset import get_data


def test_registry_loads_all_sources():
    srcs = es.expression_sources()
    assert len(srcs) > 50
    # every source has an id, a type, and at least one cancer code
    assert all(s.id and s.source_type and s.cancer_codes for s in srcs)


def test_source_types_cover_the_major_providers():
    types = {s.source_type for s in es.expression_sources()}
    for t in ("gdc", "treehouse-compendium", "recount3", "sra-ncbi-counts", "geo-matrix"):
        assert t in types

    cllmap = es.expression_source("cllmap")
    assert cllmap is not None
    assert cllmap.source_type == "geo-matrix"
    assert cllmap.source_project == "CLL-map"


def test_lookup_by_id_and_code():
    src = es.expression_source("mmrf-commpass")
    assert src is not None
    assert "MM" in src.cancer_codes
    assert src.source_type == "gdc"
    assert es.expression_source("not-a-source") is None
    assert any(s.id == "mmrf-commpass" for s in es.sources_for_cancer_code("MM"))


def test_expression_source_preserves_legacy_positional_constructor_order():
    source = es.ExpressionSource(
        "source-id",
        "expression",
        ("CODE",),
        "geo-matrix",
        "builder.py",
        ("--flag",),
        "external exemption",
        True,
        "project-id",
        ("project-a",),
        "GSE000001",
        "https://example.org/data",
        "TPM",
        1.5,
        "citation",
        "special handling",
        "SRP000001",
        "SOURCE_COHORT",
        "PolyA",
        "GEO",
        "v1",
        "primary",
        "liver",
        "pipeline",
        "notes",
        "controlled",
        "access_required",
        "https://example.org/access",
        "sample_level_annotations",
    )

    assert source.special_handling == "special handling"
    assert source.recount3_srp == "SRP000001"
    assert source.source_cohort == "SOURCE_COHORT"
    assert source.molecular_annotation_status == "sample_level_annotations"
    assert source.source_pmid is None


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


def test_mmnst_source_has_typed_ncbi_count_provenance():
    source = es.expression_source("prjna1083972-mmnst")

    assert source is not None
    assert source.cancer_codes == ("SARC_MMNST",)
    assert source.source_type == "sra-ncbi-counts"
    assert source.accession == "PRJNA1083972"
    assert source.source_cohort == "SRP493407_MMNST_2024"
    assert source.source_project == "NCBI SRA Gene Feature counts"
    assert source.unit == "NCBI Gene Feature count-derived TPM"
    assert source.tumor_origin == "primary"
    assert source.processing_pipeline


def test_vscc_source_has_typed_ncbi_count_provenance():
    source = es.expression_source("prjna994918-vscc")

    assert source is not None
    assert source.cancer_codes == ("VSCC",)
    assert source.source_type == "sra-ncbi-counts"
    assert source.accession == "PRJNA994918"
    assert source.source_cohort == "SRP449588_VSCC_2024"
    assert source.source_project == "NCBI SRA Gene Feature counts"
    assert source.unit == "NCBI Gene Feature count-derived TPM"
    assert source.tumor_origin == "mixed"
    assert source.linear_tpm_comparable
    assert not source.tpm_proxy
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
        "source_scale_class",
        "linear_tpm_comparable",
        "tpm_proxy",
        "citation",
        "source_pmid",
        "access_level",
        "acquisition_status",
        "data_access_url",
        "molecular_annotation_status",
    } <= set(df.columns)
    assert len(df) == len(es.expression_sources())


def test_structured_geo_source_pmids_match_selected_registry_provenance():
    registry = oncoref.cancer_type_registry().set_index("code")
    checked = []
    for source in es.expression_sources():
        if source.source_type != "geo-matrix" or source.source_pmid is None:
            continue
        assert re.fullmatch(r"PMID:\d+", source.source_pmid)
        for code in source.cancer_codes:
            if code not in registry.index:
                continue
            row = registry.loc[code]
            if row["source_cohort"] != source.source_cohort:
                continue
            registry_pmids = {value.strip() for value in str(row["source_pmid"]).split(";")}
            assert source.source_pmid in registry_pmids
            checked.append((source.id, code))

    assert ("gse328026-sarc-pec", "SARC_PEC") in checked


def test_pecoma_summary_availability_preserves_structured_pmid():
    source = es.expression_source("gse328026-sarc-pec")
    assert source is not None

    rows = oncoref.cancer_reference_expression_availability(
        cancer_types=["SARC_PEC"],
        normalize="tpm_clean",
        sample_qc="all",
        reference_source="summary_rows_all",
        all_sources=True,
    ).set_index("source_cohort")

    assert rows.loc[source.source_cohort, "source_pmid"] == "PMID:42331846"
    assert (
        oncoref.cancer_reference_expression_source_metadata(
            "SARC_PEC", source_cohort=source.source_cohort
        )["source_pmid"]
        == "PMID:42331846"
    )


def test_sclc_subtype_registry_sources_roundtrip_into_reference_filters():
    codes = ["SCLC_ASCL1", "SCLC_NEUROD1", "SCLC_POU2F3", "SCLC_YAP1"]
    registry = oncoref.cancer_type_registry().set_index("code")

    assert set(registry.loc[codes, "source_cohort"]) == {"SCLC_UCOLOGNE_2015"}
    for code in codes:
        source_cohort = registry.loc[code, "source_cohort"]
        availability = oncoref.cancer_reference_expression_availability(
            code,
            source_cohort=source_cohort,
            reference_source="summary_rows_all",
            sample_qc="all",
        )
        assert availability["available"].tolist() == [True]
        assert availability["source_cohort"].tolist() == [source_cohort]


def test_hcl_direct_reference_is_an_explicit_nonclassifying_scrna_proxy():
    registry = oncoref.cancer_type_registry().set_index("code").loc["HCL"]
    availability = oncoref.cancer_reference_expression_availability(
        "HCL",
        reference_source="summary_rows_all",
        sample_qc="all",
    ).iloc[0]
    coverage = oncoref.expression_reference_coverage("HCL").iloc[0]

    assert registry["source_cohort"] == "ZENODO_14917813_BOHN_2026_HCL"
    assert registry["source_pmid"] == "PMID:42200459"
    assert bool(availability["available"])
    assert availability["source_scale_class"] == "scrna_malignant_cell_pseudobulk_ntpm"
    assert not bool(availability["linear_tpm_comparable"])
    assert not bool(coverage["is_classification_target"])
    assert coverage["classification_reference_code"] is None
    assert not bool(coverage["observed_bulk_reference"])
    assert bool(coverage["single_cell_pseudobulk_reference"])
    assert coverage["expression_reference_kind"] == "single_cell_pseudobulk"
    assert coverage["consumer_recommendation"] == "reference_only"


def test_ifs_cmn_sources_keep_public_and_controlled_acquisition_explicit():
    cmn = {source.id: source for source in es.sources_for_cancer_code("CMN")}

    assert {"treehouse-ribod-25-01", "gse11482-cmn", "ega-ifs-cmn-wegert-2018"} <= set(cmn)
    assert cmn["gse11482-cmn"].unit == "TPM proxy"
    assert cmn["ega-ifs-cmn-wegert-2018"].access_level == "controlled"
    assert cmn["ega-ifs-cmn-wegert-2018"].acquisition_status == "access_required"
    assert cmn["ega-ifs-cmn-wegert-2018"].molecular_annotation_status == (
        "public_sample_level_annotations"
    )


def test_selected_physical_sources_preserve_known_availability_projects():
    availability = get_data("cancer-reference-expression-availability")
    projects_by_cohort = {
        str(source_cohort): {
            str(project)
            for project in rows["source_project"]
            if pd.notna(project) and str(project).strip()
        }
        for source_cohort, rows in availability.groupby("source_cohort", observed=True)
    }
    missing = []
    for source in es.expression_sources():
        resolution = source_matrices.resolution_for_source(source.id)
        if resolution.resolution_method != "physical_source":
            continue
        known_projects = set().union(
            *(projects_by_cohort.get(matrix.source_cohort, set()) for matrix in resolution.matrices)
        )
        if known_projects and not source.source_project:
            missing.append((source.id, sorted(known_projects)))

    assert not missing


def test_previously_missing_source_projects_use_canonical_cohort_labels():
    expected = {
        "gse114922-mds": "GEO",
        "gse118014-pannet": "GEO",
        "drmetrics-lnen-2020": "GEO (DR-metrics / Alcala LNEN)",
        "gse98894-midnet": "GEO",
        "gse32662-mtc": "GEO",
        "gse30929-lps": "GEO",
    }

    assert {
        source_id: es.expression_source(source_id).source_project for source_id in expected
    } == expected


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


def test_geo_accessions_are_structured_when_source_metadata_names_them():
    pattern = re.compile(r"\bGSE\d+\b", flags=re.IGNORECASE)

    def nested_strings(value):
        if isinstance(value, dict):
            for nested in value.values():
                yield from nested_strings(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from nested_strings(nested)
        else:
            yield str(value)

    def metadata_values(entry):
        for key, value in entry.items():
            if key in {"citation", "url"} or "file" in key:
                yield from nested_strings(value)

    for entry in es.expression_source_registry_entries():
        named = {
            match.upper() for value in metadata_values(entry) for match in pattern.findall(value)
        }
        structured = {match.upper() for match in pattern.findall(str(entry.get("accession") or ""))}
        assert named <= structured, (
            f"{entry['id']} names GEO accession(s) {sorted(named - structured)} "
            "outside its structured accession field"
        )


def test_sample_manifest_loads():
    df = samples.sample_manifest()
    assert len(df) > 1000
    for col in ("cancer_code", "source_cohort", "sample_id", "included"):
        assert col in df.columns


def test_sample_manifest_matches_availability_pipelines_and_canonical_lineages():
    manifest = samples.sample_manifest()
    availability = get_data("cancer-reference-expression-availability")
    pipeline_counts = availability.groupby("source_cohort")["processing_pipeline"].nunique()
    unique_pipeline_sources = pipeline_counts[pipeline_counts.eq(1)].index
    expected_pipeline = (
        availability[availability["source_cohort"].isin(unique_pipeline_sources)]
        .drop_duplicates("source_cohort")
        .set_index("source_cohort")["processing_pipeline"]
    )
    covered = manifest["source_cohort"].isin(expected_pipeline.index)

    assert (
        manifest.loc[covered, "processing_pipeline"]
        == manifest.loc[covered, "source_cohort"].map(expected_pipeline)
    ).all()

    lineage_labels = manifest["lineage_label"].dropna().astype(str)
    assert {resolve_cancer_type(label, strict=False) for label in lineage_labels} == set(
        lineage_labels
    )


def test_expression_source_candidates_preserve_physical_source_boundaries():
    candidates = es.expression_source_candidates()
    availability = get_data("cancer-reference-expression-availability")
    selected = availability[availability["selected"].astype(bool)].set_index("cancer_code")
    direct = candidates[candidates["source_status"].eq("direct_reference_available")].set_index(
        "cancer_code"
    )

    assert set(direct.index) == {
        "ACINIC",
        "ADCC",
        "BCC",
        "CHOL",
        "CRANIO",
        "DIPG",
        "EPN",
        "GBC",
        "HCL",
        "MENINGIOMA",
        "NEC_MERKEL",
        "SARC_IFS",
        "SARC_MMNST",
        "SARC_WDLPS",
        "SCLC_ASCL1",
        "SCLC_NEUROD1",
        "SCLC_POU2F3",
        "SCLC_YAP1",
        "VSCC",
        "cSCC",
    }
    assert set(direct.index) <= set(selected.index)
    assert (direct["reference_code"] == direct.index).all()
    assert (direct["source_cohort"] == selected.loc[direct.index, "source_cohort"]).all()
    assert (
        pd.to_numeric(direct["estimated_samples"])
        == pd.to_numeric(selected.loc[direct.index, "n_reference_samples"])
    ).all()

    assert direct.loc["ADCC", "accession"] == "GSE294016"
    assert direct.loc["ACINIC", "accession"] == "GSE294016"
    assert direct.loc["NEC_MERKEL", "accession"] == "GSE235092"
    assert direct.loc["HCL", "accession"] == "DOI:10.5281/zenodo.14917813"
    assert int(direct.loc["HCL", "estimated_samples"]) == 5
    assert int(direct.loc["BCC", "estimated_samples"]) == 25
    assert int(direct.loc["cSCC", "estimated_samples"]) == 10
    assert int(direct.loc["GBC", "estimated_samples"]) == 10
    assert int(direct.loc["EPN", "estimated_samples"]) == 11
    assert int(direct.loc["CRANIO", "estimated_samples"]) == 29
    assert int(direct.loc["DIPG", "estimated_samples"]) == 32
    assert int(direct.loc["VSCC", "estimated_samples"]) == 9
    assert direct.loc["EPN", "accession"] == "GSE141460"
    assert direct.loc["CRANIO", "accession"] == "OpenPBTA release-v23-20230115"
    assert direct.loc["DIPG", "accession"] == "OpenPBTA release-v23-20230115"
    assert "no papillary tumors" in direct.loc["CRANIO", "notes"]
    assert "no per-donor BRAF V600E call is inferred" in direct.loc["HCL", "notes"]
    assert direct.loc["CHOL", "source_cohort"] == "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES"

    divergent = candidates.set_index("cancer_code").loc[["FL", "NPC", "SARC_ASPS", "SARC_MYXLPS"]]
    assert not divergent["source_status"].eq("direct_reference_available").any()
    assert (divergent["source_cohort"] != selected.loc[divergent.index, "source_cohort"]).all()


def test_nci_gap_direct_references_have_explicit_scale_and_target_policy():
    expected = {
        "BCC": ("GSE125285_BCC_CSCC", "bulk_rnaseq_cpm_proxy", False, 25),
        "cSCC": ("GSE125285_BCC_CSCC", "bulk_rnaseq_cpm_proxy", False, 10),
        "GBC": ("GSE139682_GBC", "linear_rnaseq_tpm", True, 10),
        "CRANIO": ("OPENPBTA_V23_CBTN_CRANIO", "linear_rnaseq_tpm", True, 29),
        "DIPG": ("OPENPBTA_V23_DIPG_H3K27", "linear_rnaseq_tpm", True, 32),
        "VSCC": ("SRP449588_VSCC_2024", "ncbi_gene_feature_count_tpm", True, 9),
        "EPN": (
            "GSE141460_GOJO_2020_EPN",
            "scrna_malignant_cell_mean_tpm_pseudobulk",
            False,
            11,
        ),
    }
    registry = oncoref.cancer_type_registry().set_index("code")
    manifest = samples.sample_manifest()
    for code, (cohort, scale, comparable, n_samples) in expected.items():
        availability = oncoref.cancer_reference_expression_availability(
            code,
            reference_source="summary_rows_all",
            sample_qc="all",
        ).iloc[0]
        coverage = oncoref.expression_reference_coverage(code).iloc[0]
        included = manifest[
            manifest["source_cohort"].eq(cohort)
            & manifest["cancer_code"].eq(code)
            & manifest["included"].astype(str).str.lower().isin(("true", "1"))
        ]

        assert registry.loc[code, "source_cohort"] == cohort
        assert availability["source_scale_class"] == scale
        assert bool(availability["linear_tpm_comparable"]) is comparable
        assert int(availability["n_reference_samples"]) == n_samples
        assert len(included) == n_samples
        assert not bool(coverage["is_classification_target"])
        assert coverage["consumer_recommendation"] == "reference_only"


def test_nci_gap_candidates_have_explicit_non_promoting_owner_decisions():
    expected_status = {
        "VAGC": "deferred_no_dedicated_source",
        "FTC": "deferred_pooled_source_not_direct",
        "PPC": "deferred_pooled_source_not_direct",
        "PENSCC": "microarray_proxy_only",
        "URETH": "deferred_no_dedicated_source",
        "ANSC": "bulk_candidate_needs_sample_selection",
        "PITNET": "bulk_candidate_requires_sra_build",
    }
    candidates = es.expression_source_candidates()
    rows = candidates[candidates["cancer_code"].isin(expected_status)].set_index("cancer_code")

    assert len(rows) == len(expected_status)
    assert rows["source_status"].to_dict() == expected_status
    for column in (
        "source_status",
        "assay",
        "source_scope",
        "processing_plan",
        "gene_id_plan",
        "normalization_plan",
        "notes",
    ):
        assert rows[column].notna().all()
        assert rows[column].astype(str).str.strip().ne("").all()
    assert set(rows.loc[["FTC", "PPC"], "reference_code"]) == {"OV"}
    assert not rows["source_status"].eq("direct_reference_available").any()
    assert not (set(rows.index) & set(source_matrices.registry()["cancer_code"].astype(str)))
    assert int(rows.loc["ANSC", "estimated_samples"]) == 23
    assert "lacks an explicit public matrix-to-GEO sample crosswalk" in rows.loc["ANSC", "notes"]


def test_new_expression_gap_candidates_have_reproducible_dispositions():
    expected = {
        "MENINGIOMA": ("direct_reference_available", "GSE270638", 384),
        "PITNET": ("bulk_candidate_requires_sra_build", "GSE209903", 65),
        "CHOROID_PLEXUS": ("microarray_proxy_only", "GSE60892", 40),
        "ALCL": ("microarray_proxy_only", "GSE58445", 79),
    }
    rows = es.expression_source_candidates()
    rows = rows[rows["cancer_code"].isin(expected)].set_index("cancer_code")

    assert set(rows.index) == set(expected)
    for code, (status, accession, n_samples) in expected.items():
        row = rows.loc[code]
        assert row["source_status"] == status
        assert row["accession"] == accession
        assert int(row["estimated_samples"]) == n_samples
        assert str(row["processing_plan"]).strip()
        assert str(row["gene_id_plan"]).strip()
        assert str(row["normalization_plan"]).strip()
        assert str(row["notes"]).strip()

    meningioma = es.expression_source("gse270638-meningioma")
    assert meningioma is not None
    assert meningioma.builder == "scripts/build_geo_matrix.py"
    assert meningioma.cancer_codes == ("MENINGIOMA",)
    assert meningioma.source_cohort == "GSE270638_MENINGIOMA_2024"
    assert meningioma.source_scale_class == "linear_rnaseq_tpm"
    assert meningioma.linear_tpm_comparable is True
    assert meningioma.tpm_proxy is False

    unavailable = {"PITNET", "CHOROID_PLEXUS", "ALCL"}
    availability = oncoref.cancer_reference_expression_availability(
        sorted(unavailable), reference_source="summary_rows_all", sample_qc="all"
    )
    assert set(availability["requested_code"]) == unavailable
    assert not availability["available"].any()


def test_non_testicular_gct_candidates_record_proxy_and_rejection_boundaries():
    codes = {
        "GCT_OV_YST",
        "GCT_OV_IMT",
        "GCT_OV_DYS",
        "GCT_CNS_GER",
        "GCT_CNS_NGGCT",
    }
    rows = es.expression_source_candidates()
    rows = rows[rows["cancer_code"].isin(codes)].set_index("cancer_code")

    assert set(rows.index) == codes
    assert rows["accession"].to_dict() == {
        "GCT_OV_YST": "GSE169733",
        "GCT_OV_IMT": "GSE229343",
        "GCT_OV_DYS": "DOI:10.1038/s41416-025-03012-6",
        "GCT_CNS_GER": "GSE19348",
        "GCT_CNS_NGGCT": "GSE19348",
    }
    assert int(rows.loc["GCT_OV_YST", "estimated_samples"]) == 10
    assert int(rows.loc["GCT_OV_IMT", "estimated_samples"]) == 3
    assert int(rows.loc["GCT_OV_DYS", "estimated_samples"]) == 1
    assert "GSE247518" in rows.loc["GCT_OV_YST", "notes"]
    assert "GSE156170" in rows.loc["GCT_OV_IMT", "notes"]
    assert (
        rows.loc[["GCT_CNS_GER", "GCT_CNS_NGGCT"], "source_status"]
        .eq("microarray_proxy_only")
        .all()
    )
    assert not rows["reference_code"].fillna("").eq("TGCT").any()
    assert not (codes & set(source_matrices.registry()["cancer_code"].astype(str)))

    availability = oncoref.cancer_reference_expression_availability(sorted(codes))
    assert set(availability["requested_code"]) == codes
    assert not availability["available"].any()


def test_samples_for_cancer_code_included_only():
    inc = samples.samples_for_cancer_code("BL", included_only=True)
    allrows = samples.samples_for_cancer_code("BL", included_only=False)
    assert len(inc) <= len(allrows)
    assert (inc["included"].astype(str).str.lower() == "true").all()


def test_mmnst_sample_manifest_preserves_controls_without_routing_them():
    cohort = samples.samples_for_cohort("SRP493407_MMNST_2024", included_only=False)
    included = samples.samples_for_cancer_code("SARC_MMNST")

    assert len(cohort) == 6
    assert included["sample_id"].tolist() == ["SRR28227826", "SRR28227825", "SRR28227824"]
    controls = cohort.loc[~cohort["included"].astype(bool)]
    assert controls["sample_id"].tolist() == ["SRR28227823", "SRR28227822", "SRR28227821"]
    assert set(controls["exclusion_reason"]) == {
        "independent_normal_control_excluded_from_tumor_reference"
    }


def test_sample_counts_sum():
    counts = samples.sample_counts_by_cancer_code()
    assert counts.sum() > 1000
    assert (counts > 0).all()
