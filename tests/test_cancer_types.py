# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd
import pytest

import oncoref as cd
from oncoref import cancer_types
from oncoref.load_dataset import get_data

HPA_RNA_V23_TISSUES = {
    "adipose tissue",
    "adrenal gland",
    "amygdala",
    "appendix",
    "basal ganglia",
    "bone marrow",
    "breast",
    "cerebellum",
    "cerebral cortex",
    "cervix",
    "choroid plexus",
    "colon",
    "duodenum",
    "endometrium",
    "epididymis",
    "esophagus",
    "fallopian tube",
    "gallbladder",
    "heart muscle",
    "hippocampal formation",
    "hypothalamus",
    "kidney",
    "liver",
    "lung",
    "lymph node",
    "midbrain",
    "ovary",
    "pancreas",
    "parathyroid gland",
    "pituitary gland",
    "placenta",
    "prostate",
    "rectum",
    "retina",
    "salivary gland",
    "seminal vesicle",
    "skeletal muscle",
    "skin",
    "small intestine",
    "smooth muscle",
    "spinal cord",
    "spleen",
    "stomach",
    "testis",
    "thymus",
    "thyroid gland",
    "tongue",
    "tonsil",
    "urinary bladder",
    "vagina",
}


def test_resolve_common_name_alias():
    assert cancer_types.resolve_cancer_type("prostate") == "PRAD"
    assert cancer_types.resolve_cancer_type("melanoma") == "SKCM"
    assert cancer_types.resolve_cancer_type("AciCC") == "ACINIC"
    assert cancer_types.resolve_cancer_type("ACICC") == "ACINIC"


def test_resolve_mplps_aliases_without_collapsing_other_liposarcomas():
    assert cancer_types.resolve_cancer_type("MPLPS") == "SARC_MPLPS"
    assert cancer_types.resolve_cancer_type("myxoid pleomorphic liposarcoma") == "SARC_MPLPS"
    assert cancer_types.resolve_cancer_type("pleomorphic myxoid liposarcoma") == "SARC_MPLPS"
    assert cancer_types.resolve_cancer_type("myxoid liposarcoma") == "SARC_MYXLPS"
    assert cancer_types.resolve_cancer_type("pleomorphic liposarcoma") == "SARC_PLEOLPS"


def test_resolve_canonical_code_passthrough():
    assert cancer_types.resolve_cancer_type("PRAD") == "PRAD"


def test_resolve_renamed_code():
    # Pre-rename codes resolve to the current canonical code.
    assert cancer_types.resolve_cancer_type("OS") == "SARC_OS"
    assert cancer_types.resolve_cancer_type("PANNET") == "NET_PANCREAS"


def test_resolve_display_name_case_insensitive():
    name = cd.CANCER_TYPE_NAMES["PRAD"]
    assert cancer_types.resolve_cancer_type(name.lower()) == "PRAD"


def test_resolve_none_passthrough():
    assert cancer_types.resolve_cancer_type(None) is None


def test_resolve_unknown_raises_strict():
    with pytest.raises(ValueError):
        cancer_types.resolve_cancer_type("not_a_real_cancer")


def test_resolve_unknown_nonstrict_returns_none():
    assert cancer_types.resolve_cancer_type("not_a_real_cancer", strict=False) is None


def test_canonical_cancer_code_is_pure():
    assert cancer_types.canonical_cancer_code("MID_NET") == "NET_MIDGUT"
    assert cancer_types.canonical_cancer_code("PRAD") == "PRAD"
    assert cancer_types.canonical_cancer_code("anything_else") == "anything_else"


def test_registry_has_core_columns():
    df = cancer_types.cancer_type_registry()
    for col in (
        "code",
        "name",
        "family",
        "primary_tissue",
        "parent_code",
        "reference_source",
        "classification_reference_code",
        "is_classification_target",
    ):
        assert col in df.columns
    records = df.set_index("code")
    assert records.loc["CRC_MSI", "reference_source"] == "member_union"
    assert records.loc["CRC_MSI", "classification_reference_code"] == "CRC_MSI"
    assert bool(records.loc["CRC_MSI", "is_classification_target"]) is True
    assert records.loc["STAD_MSI", "reference_source"] == "own_cohort"
    assert records.loc["STAD_MSI", "classification_reference_code"] == "STAD_MSI"
    assert bool(records.loc["STAD_MSI", "is_classification_target"]) is True
    assert "PRAD" in set(df["code"])


def test_cancer_type_info_assembles_derived_fields():
    info = cancer_types.cancer_type_info("prostate")
    assert info["code"] == "PRAD"
    assert info["name"]
    # derived fields pulled from the TMB + burden tables (cycle-safe lazy import)
    assert "burden_category" in info
    assert "tmb" in info
    assert info["burden_category"] == "prostate"


def test_cancer_type_info_none_passthrough():
    assert cancer_types.cancer_type_info(None) is None


def test_synonyms_roundtrip():
    syns = cancer_types.cancer_type_synonyms("PRAD")
    assert "prostate" in syns
    assert "PRAD" not in syns  # the code itself is excluded


def test_families_nonempty():
    fams = cancer_types.cancer_type_families()
    assert fams
    assert all(isinstance(v, str) and v for v in fams.values())


def test_cohort_aggregates_sarc_grand_union():
    members = cancer_types.cohort_aggregate_members("SARC")
    assert members
    assert "SARC" not in members  # no self-membership


def test_cohort_registry_is_validation_authority():
    ids = cancer_types.known_cohort_ids()
    assert ids
    assert isinstance(ids, frozenset)


def test_treehouse_tcga_cohort_uses_canonical_identity_and_exact_legacy_alias():
    canonical = "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES"
    legacy = "TREEHOUSE_POLYA_25_01_TCGA_SUBSET"
    derived = "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY"
    registry = cancer_types.cohort_registry_df().set_index("cohort_id")
    matrices = get_data("source-matrices")
    availability = get_data("cancer-reference-expression-availability")
    canonical_matrices = matrices.loc[matrices["source_cohort"] == canonical]
    canonical_sources = availability.loc[availability["source_cohort"] == canonical]

    assert canonical in cancer_types.known_cohort_ids()
    assert legacy not in cancer_types.known_cohort_ids()
    assert registry.loc[canonical, "n_samples"] == 9541
    assert registry.loc[canonical, "n_codes"] == 32
    assert len(canonical_matrices) == 32
    assert canonical_matrices["n_samples"].sum() == 9541
    assert len(canonical_sources) == 32
    assert canonical_sources["n_reference_samples"].sum() == 9541
    assert derived in registry.index
    derived_sources = availability.loc[availability["source_cohort"] == derived]
    assert set(derived_sources["cancer_code"]) == {
        "SARC_DDLPS",
        "SARC_PLEOLPS",
        "SARC_WDLPS",
    }
    assert derived_sources["n_reference_samples"].sum() == 55
    with pytest.warns(DeprecationWarning, match="TCGA_SUBSET"):
        assert cancer_types.resolve_cohort_id(legacy) == canonical
    assert cancer_types.canonical_cohort_id(f"{legacy}_DERIVED") == f"{legacy}_DERIVED"


def test_computed_cohort_registry_members_are_derived_from_live_aggregates():
    registry = cancer_types.cohort_registry_df().set_index("cohort_id")
    shipped_registry = get_data("cohort-registry").set_index("cohort_id")
    for code, cohort_id in (
        ("SARC", "COMPUTED_PAN_SARCOMA"),
        ("CRC", "COMPUTED_COLORECTAL"),
    ):
        expected = cancer_types.cohort_aggregate_members(code)
        row = registry.loc[cohort_id]
        members = str(row["member_cohorts"]).split(";")
        assert members == expected
        assert int(row["n_codes"]) == len(expected)

        shipped_row = shipped_registry.loc[cohort_id]
        assert str(shipped_row["member_cohorts"]).split(";") == expected
        assert int(shipped_row["n_codes"]) == len(expected)

    sarcoma_members = set(cancer_types.cohort_aggregate_members("SARC"))
    assert {"SARC_MMNST", "SARC_MPLPS"} <= sarcoma_members


def test_source_cohort_counts_are_derived_from_expression_registry(monkeypatch):
    shipped = get_data("cohort-registry")
    stale = shipped.copy()
    matching = stale["cohort_id"].eq("GSE294016_BARTL_2025_SGC")
    stale.loc[matching, ["n_samples", "n_codes"]] = [123, 1]
    original_get_data = cancer_types.get_data

    def get_data_with_stale_cohort_registry(name, *args, **kwargs):
        if name == "cohort-registry":
            return stale.copy()
        return original_get_data(name, *args, **kwargs)

    monkeypatch.setattr(cancer_types, "get_data", get_data_with_stale_cohort_registry)
    cancer_types._expression_source_cohort_counts.cache_clear()

    row = cancer_types.cohort_registry_df().set_index("cohort_id").loc["GSE294016_BARTL_2025_SGC"]
    assert int(row["n_samples"]) == 95
    assert int(row["n_codes"]) == 2


def test_source_cohort_counts_require_explicit_boolean_for_overlapping_routes(monkeypatch):
    from oncoref import expression_registry

    source = {
        "id": "overlap-test",
        "source_cohort": "OVERLAP_TEST",
        "cancer_codes": ["PARENT", "CHILD"],
        "expected_source_samples": 5,
        "expected_samples_by_code": {"PARENT": 5, "CHILD": 5},
    }
    monkeypatch.setattr(
        expression_registry,
        "expression_source_registry_entries",
        lambda: (source,),
    )
    cancer_types._expression_source_cohort_counts.cache_clear()
    with pytest.raises(ValueError, match="routes more samples than it contains"):
        cancer_types._expression_source_cohort_counts()

    source["routed_samples_may_overlap"] = "true"
    cancer_types._expression_source_cohort_counts.cache_clear()
    with pytest.raises(ValueError, match="must be a YAML boolean"):
        cancer_types._expression_source_cohort_counts()

    source["routed_samples_may_overlap"] = True
    cancer_types._expression_source_cohort_counts.cache_clear()
    counts = cancer_types._expression_source_cohort_counts()
    assert counts["OVERLAP_TEST"]["physical_samples"] == 5
    cancer_types._expression_source_cohort_counts.cache_clear()


def test_cohort_source_version_parses_ensembl_release():
    # The per-cohort source_version for auditing the canonical gene-ID space: a code
    # resolves to its source cohort, whose provenance records the harmonized Ensembl
    # release. The shipped cohorts are harmonized to Ensembl 112.
    assert cancer_types.cohort_source_version("LUAD") == "112"
    assert cancer_types.cohort_source_version("ACC") == "112"
    # an unknown code has no recorded source version (rather than raising)
    assert cancer_types.cohort_source_version("NOT_A_REAL_CODE") is None


def test_clear_caches_allows_reload(monkeypatch):
    # The names view caches the registry; clearing forces a re-read.
    assert "PRAD" in cd.CANCER_TYPE_NAMES
    cancer_types._clear_caches()
    assert "PRAD" in cd.CANCER_TYPE_NAMES


def test_every_registry_family_has_a_curated_display_name():
    # Drift guard: every registry family must have a curated label, not the
    # title-cased fallback. (This caught the stale 'cns'/'endocrine' keys left
    # after the registry split those into cns-*/endocrine-* families.)
    families = set(cancer_types.cancer_type_registry()["family"].dropna().astype(str))
    missing = sorted(f for f in families if f not in cancer_types._FAMILY_DISPLAY_NAMES)
    assert not missing, f"registry families without a curated display name: {missing}"


def test_lineage_group_resolution():
    # Coarse histogenesis rollup: family default + per-code override (inherited).
    assert cancer_types.cancer_lineage_group("LUAD") == "Epithelial"
    assert cancer_types.cancer_lineage_group("SARC_OS") == "Sarcoma"
    assert cancer_types.cancer_lineage_group("SKCM") == "Melanoma"
    assert cancer_types.cancer_lineage_group("NET_PANCREAS") == "Neuroendocrine"
    assert cancer_types.cancer_lineage_group("ASTB") == "CNS"
    # NBL overrides its neuroendocrine family default -> Embryonal, inherited by subtypes.
    assert cancer_types.cancer_lineage_group("NBL") == "Embryonal"
    assert cancer_types.cancer_lineage_group("NBL_MYCNamp") == "Embryonal"
    assert cancer_types.cancer_lineage_group("not_a_real_cancer") is None


def test_nsclc_is_lung_histology_parent():
    assert cancer_types.resolve_cancer_type("nsclc") == "NSCLC"
    assert cancer_types.resolve_cancer_type("non small cell lung cancer") == "NSCLC"
    assert cancer_types.cancer_type_subtypes_of("NSCLC") == ["LUAD", "LUSC"]
    assert cancer_types.cancer_type_ancestors("LUAD") == ["NSCLC"]
    assert cancer_types.cancer_type_ancestors("LUSC") == ["NSCLC"]
    assert cancer_types.cancer_type_ancestors("LUAD_EGFR") == ["LUAD", "NSCLC"]

    records = cancer_types.cancer_type_records(under="NSCLC")
    assert records["code"].tolist() == [
        "NSCLC",
        "LUAD",
        "LUAD_EGFR",
        "LUAD_KRAS",
        "LUAD_STK11",
        "LUSC",
    ]
    registry = cancer_types.cancer_type_registry().set_index("code")
    assert bool(registry.loc["NSCLC", "mixture_cohort"]) is True


def test_astb_registry_row_for_trufflepig_parity():
    assert cancer_types.resolve_cancer_type("ASTB") == "ASTB"
    assert cancer_types.resolve_cancer_type("astroblastoma (mn1-altered)") == "ASTB"

    raw = cancer_types.cancer_type_registry().set_index("code").loc["ASTB"]
    assert bool(raw["pediatric"]) is True
    assert raw["fusion_driven"] == "defining"
    assert raw["fusion_driver"] == "MN1-BEND2"

    row = cancer_types.cancer_type_records(["ASTB"]).iloc[0]
    assert row["name"] == "Astroblastoma (MN1-altered)"


def test_mmnst_registry_row_for_expression_reference():
    assert cancer_types.resolve_cancer_type("SARC_MMNST") == "SARC_MMNST"
    assert cancer_types.cancer_lineage_group("SARC_MMNST") == "Sarcoma"

    raw = cancer_types.cancer_type_registry().set_index("code").loc["SARC_MMNST"]
    assert raw["parent_code"] == "SARC"
    assert raw["primary_tissue"] == "nerve_sheath"
    assert raw["expression_source"] == "NCBI SRA Gene Feature counts"
    assert raw["source_cohort"] == "SRP493407_MMNST_2024"
    assert raw["source_pmid"] == "PMID:41995777"
    assert raw["fusion_driven"] == "none"

    row = cancer_types.cancer_type_records(["SARC_MMNST"]).iloc[0]
    assert row["name"] == "Malignant Melanotic Nerve Sheath Tumor"
    assert row["family"] == "sarcoma"
    assert row["primary_tissue"] == "nerve_sheath"
    assert row["source_cohort"] == "SRP493407_MMNST_2024"
    assert row["source_pmid"] == "PMID:41995777"
    assert row["lineage_group"] == "Sarcoma"
    assert row["normal_tissue_code"] == "soft_tissue"
    assert row["hpa_tissues"] == ()
    assert bool(row["has_expression_matrix"]) is True


def test_mplps_registry_and_reference_gap_are_explicit():
    raw = cancer_types.cancer_type_registry().set_index("code").loc["SARC_MPLPS"]
    assert bool(raw["pediatric"]) is True
    assert raw["grade_tier"] == "high"
    assert raw["fusion_driven"] == "none"
    assert "lacks DDIT3 rearrangement and MDM2 amplification" in raw["notes"]
    assert "RB1/13q14 loss" in raw["notes"]

    row = cancer_types.cancer_type_records(["SARC_MPLPS"]).iloc[0]
    assert row["parent_code"] == "SARC_LPS"
    assert row["ontology_kind"] == "histologic_type"
    assert row["reference_source"] == "parent"
    assert row["classification_reference_code"] == "SARC_LPS"
    assert bool(row["is_classification_target"]) is False
    assert bool(row["has_expression_matrix"]) is False


def test_unspecified_liposarcoma_is_a_source_bucket_not_a_tumor_entity():
    row = cancer_types.cancer_type_records(["SARC_LPS_UNSPEC"]).iloc[0]
    assert row["ontology_level"] == "evidence_scope"
    assert row["ontology_kind"] == "source_scope"
    assert row["reference_source"] == "none"
    assert row["classification_reference_code"] == "SARC_LPS"
    assert bool(row["is_classification_target"]) is False
    assert bool(row["has_expression_matrix"]) is True

    coverage = cancer_types.expression_reference_coverage(
        ["SARC_LPS", "SARC_MPLPS", "SARC_LPS_UNSPEC"]
    ).set_index("code")
    assert coverage.loc["SARC_LPS", "computed_expression_member_codes"] == (
        "SARC_DDLPS",
        "SARC_WDLPS",
        "SARC_MYXLPS",
        "SARC_PLEOLPS",
        "SARC_LPS_UNSPEC",
    )
    assert not bool(coverage.loc["SARC_MPLPS", "has_expression_reference"])
    assert coverage.loc["SARC_MPLPS", "consumer_recommendation"] == "parent_reference"
    assert not bool(coverage.loc["SARC_LPS_UNSPEC", "has_direct_expression_reference"])


def test_every_registry_family_rolls_up_to_a_lineage_group():
    # Drift guard: the coarse rollup must cover every registry family, so a new
    # family can't silently yield a None lineage group.
    families = set(cancer_types.cancer_type_registry()["family"].dropna().astype(str))
    groups = cancer_types.cancer_lineage_groups()
    missing = sorted(f for f in families if f not in groups)
    assert not missing, f"registry families with no lineage group: {missing}"


def test_cancer_type_records_query_hierarchy_and_molecular_groups():
    crc = cancer_types.cancer_type_records(under="CRC")
    assert crc["code"].tolist() == [
        "CRC",
        "CRC_MSI",
        "COAD",
        "READ",
        "COAD_MSI",
        "COAD_MSS",
        "READ_MSI",
        "READ_MSS",
    ]
    assert cancer_types.cancer_type_codes(subtype_group="MSI", under="CRC") == [
        "CRC_MSI",
        "COAD_MSI",
        "READ_MSI",
    ]
    epithelial_msi = cancer_types.cancer_type_records(
        subtype_group="MSI", lineage_group="Epithelial"
    )
    assert {"COAD_MSI", "READ_MSI", "UCEC_MSI"} <= set(epithelial_msi["code"])
    assert set(epithelial_msi["lineage_group"]) == {"Epithelial"}
    assert cancer_types.cancer_type_records(ontology_level=[]).empty
    assert cancer_types.cancer_type_records(subtype_group=[]).empty


def test_cancer_type_records_empty_selection_stays_empty():
    empty = cancer_types.cancer_type_records([])
    assert empty.empty
    assert list(empty.columns) == list(cancer_types.cancer_type_records(["PRAD"]).columns)
    assert cancer_types.cancer_type_records([], under="CRC").empty
    assert cancer_types.cancer_type_siblings("CRC").empty


def test_cancer_type_path_makes_semantic_levels_explicit():
    path = cancer_types.cancer_type_path("COAD_MSI")
    assert path[["kind", "code"]].apply(tuple, axis=1).tolist() == [
        ("lineage_group", "Epithelial"),
        ("family", "carcinoma-gi"),
        ("cancer_type", "CRC"),
        ("cancer_type", "COAD"),
        ("cancer_type", "COAD_MSI"),
    ]
    assert path.iloc[-1]["normal_tissue_code"] == "colon"

    crc_msi = cancer_types.cancer_type_path("CRC_MSI")
    assert crc_msi[["kind", "code"]].apply(tuple, axis=1).tolist() == [
        ("lineage_group", "Epithelial"),
        ("family", "carcinoma-gi"),
        ("cancer_type", "CRC"),
        ("cancer_type", "CRC_MSI"),
    ]


def test_cancer_type_records_expose_explicit_ontology_levels():
    records = cancer_types.cancer_type_records(
        ["CRC", "CRC_MSI", "OV", "FTC", "PPC", "ALCL", "ATC", "PMBCL", "PCN"]
    ).set_index("code")

    assert records.loc["CRC", "ontology_level"] == "grouping"
    assert records.loc["CRC", "ontology_kind"] == "computed_union"
    assert records.loc["CRC_MSI", "parent_code"] == "CRC"
    assert records.loc["CRC_MSI", "ontology_level"] == "molecular_subtype"
    assert records.loc["CRC_MSI", "ontology_kind"] == "molecular_source_scope"

    assert bool(cancer_types.is_mixture_cohort("OV")) is True
    assert records.loc["OV", "ontology_level"] == "grouping"
    assert records.loc["OV", "ontology_kind"] == "anatomic_group"
    assert records.loc["FTC", "ontology_level"] == "type"
    assert records.loc["FTC", "ontology_kind"] == "anatomic_type"
    assert records.loc["PPC", "ontology_level"] == "type"
    assert records.loc["PPC", "ontology_kind"] == "anatomic_type"

    assert records.loc["ALCL", "ontology_level"] == "type"
    assert records.loc["ALCL", "ontology_kind"] == "primary_type"
    assert records.loc["ATC", "ontology_level"] == "type"
    assert records.loc["ATC", "ontology_kind"] == "differentiation_type"
    assert records.loc["PMBCL", "ontology_level"] == "type"
    assert records.loc["PMBCL", "ontology_kind"] == "histologic_type"
    assert records.loc["PCN", "ontology_level"] == "type"
    assert records.loc["PCN", "ontology_kind"] == "clinical_type"

    assert cancer_types.cancer_type_codes(
        under="CRC", ontology_level="molecular_subtype", ontology_kind="molecular_source_scope"
    ) == ["CRC_MSI"]

    sarcoma = cancer_types.cancer_type_records(
        ["SARC", "SARC_OS", "SARC_RMS", "SARC_RMS_ARMS"]
    ).set_index("code")
    assert sarcoma.loc["SARC", "ontology_level"] == "grouping"
    assert sarcoma.loc["SARC_OS", "ontology_level"] == "type"
    assert sarcoma.loc["SARC_RMS", "ontology_level"] == "type"
    assert sarcoma.loc["SARC_RMS", "ontology_kind"] == "computed_union"
    assert sarcoma.loc["SARC_RMS_ARMS", "ontology_level"] == "molecular_subtype"
    assert sarcoma.loc["SARC_RMS_ARMS", "ontology_kind"] == "fusion_molecular_subtype"


def test_computed_union_and_evidence_scope_semantics_are_consistent():
    records = cancer_types.cancer_type_records().set_index("code")
    computed_codes = set(records.index[records["expression_source"].astype(str) == "computed"])
    computed_kind_codes = set(records.index[records["ontology_kind"] == "computed_union"])

    assert computed_codes == computed_kind_codes
    assert cancer_types.computed_union_codes() == [
        "CRC",
        "RCC",
        "SARC",
        "THYM_EPITHELIAL",
        "NEN",
        "NET",
        "NEC",
        "NEC_LUNG",
        "SARC_RMS",
        "SARC_LPS",
        "SARC_ESS",
    ]
    assert cd.computed_union_codes() == cancer_types.computed_union_codes()
    assert cd.cancer_ontology.computed_union_codes() == cancer_types.computed_union_codes()

    evidence_scopes = records.loc[["NET_NONPANCREATIC", "NEN_EXTRAPULMONARY_HG"]]
    assert set(evidence_scopes["ontology_level"]) == {"evidence_scope"}
    assert set(evidence_scopes["ontology_kind"]) == {"source_scope"}
    assert set(evidence_scopes["is_classification_target"]) == {False}
    assert not {"NET_NONPANCREATIC", "NEN_EXTRAPULMONARY_HG"} & set(
        cancer_types.cancer_type_codes(ontology_level="grouping")
    )


def test_cancer_type_category_schema_and_summary_are_public_contract():
    records = cancer_types.cancer_type_records()
    schema = cancer_types.cancer_type_category_schema()

    assert list(schema.columns) == [
        "dimension",
        "value",
        "description",
        "n_codes",
        "example_codes",
        "is_reportable_reference",
    ]
    assert set(records["ontology_level"]) <= set(cancer_types.ONTOLOGY_LEVEL_VALUES)
    assert schema[schema["dimension"] == "ontology_level"]["value"].tolist() == list(
        cancer_types.ONTOLOGY_LEVEL_VALUES
    )
    assert schema[schema["dimension"] == "reference_source"]["value"].tolist() == list(
        cancer_types.REFERENCE_SOURCE_VALUES
    )
    ref_rows = schema[schema["dimension"] == "reference_source"].set_index("value")
    assert bool(ref_rows.loc["own_cohort", "is_reportable_reference"]) is True
    assert bool(ref_rows.loc["member_union", "is_reportable_reference"]) is True
    assert bool(ref_rows.loc["parent", "is_reportable_reference"]) is False
    assert bool(ref_rows.loc["none", "is_reportable_reference"]) is False
    assert ref_rows.loc["own_cohort", "n_codes"] == len(
        records[records["reference_source"] == "own_cohort"]
    )
    assert ref_rows.loc["member_union", "example_codes"]

    summary = cancer_types.cancer_type_category_summary()
    assert list(summary.columns) == [
        "ontology_level",
        "ontology_kind",
        "reference_source",
        "n_codes",
        "n_classification_targets",
        "n_expression_matrices",
        "example_codes",
    ]
    assert summary["n_codes"].sum() == len(records)
    assert summary["n_classification_targets"].sum() == int(
        records["is_classification_target"].sum()
    )
    assert summary["n_expression_matrices"].sum() == int(records["has_expression_matrix"].sum())
    crc_msi_combo = summary[
        (summary["ontology_level"] == "molecular_subtype")
        & (summary["ontology_kind"] == "molecular_source_scope")
        & (summary["reference_source"] == "member_union")
    ]
    assert "CRC_MSI" in crc_msi_combo.iloc[0]["example_codes"]

    assert cd.ONTOLOGY_LEVEL_VALUES == cancer_types.ONTOLOGY_LEVEL_VALUES
    assert (
        cd.cancer_ontology.cancer_type_category_summary()["n_codes"].sum()
        == summary["n_codes"].sum()
    )


def test_reference_source_and_review_policy_drive_classification_targets():
    records = cancer_types.cancer_type_records(
        [
            "CRC_MSI",
            "COAD_MSI",
            "READ_MSI",
            "NEN_EXTRAPULMONARY_HG",
            "NET_NONPANCREATIC",
            "NEN",
            "NET",
            "NEC",
            "SARC",
            "SARC_LPS",
            "STAD_MSI",
            "CRC",
            "NSCLC",
            "OV",
            "BTC",
            "SGC",
            "RCC",
            "RCC_NCC",
            "RCC_NCC_UNCLASSIFIED",
            "THYM_EPITHELIAL",
            "THYMCA",
            "NEC_LUNG",
        ]
    ).set_index("code")

    assert set(records.loc[records["reference_source"] == "own_cohort"].index) == {
        "COAD_MSI",
        "READ_MSI",
        "STAD_MSI",
        "OV",
    }
    assert set(records.loc[records["reference_source"] == "member_union"].index) == {
        "CRC_MSI",
        "NEN",
        "NET",
        "NEC",
        "SARC",
        "SARC_LPS",
        "CRC",
        "NSCLC",
        "BTC",
        "SGC",
        "RCC",
        "THYM_EPITHELIAL",
        "NEC_LUNG",
    }
    assert set(records.loc[records["reference_source"] == "parent"].index) == {
        "RCC_NCC_UNCLASSIFIED",
        "THYMCA",
    }
    assert set(records.loc[records["reference_source"] == "none"].index) == {
        "RCC_NCC",
        "NET_NONPANCREATIC",
        "NEN_EXTRAPULMONARY_HG",
    }

    non_targets = {
        "RCC_NCC",
        "RCC_NCC_UNCLASSIFIED",
        "SGC",
        "THYMCA",
        "NET_NONPANCREATIC",
        "NEN_EXTRAPULMONARY_HG",
    }
    assert not records.loc[list(non_targets), "is_classification_target"].any()
    assert (
        set(records.index[records["is_classification_target"]]) == set(records.index) - non_targets
    )
    with pytest.raises(ValueError, match="classification_target"):
        cancer_types.cancer_type_codes(classification_target="maybe")

    for code in ["CRC_MSI", "SARC", "CRC", "NET", "NSCLC", "OV", "BTC"]:
        assert bool(records.loc[code, "is_classification_target"]) is True
        assert cancer_types.is_classification_target(code) is True

    for code in non_targets:
        assert cancer_types.is_classification_target(code) is False
        assert code not in cancer_types.classification_target_codes()

    reference_backed_codes = cancer_types.cancer_type_codes(
        reference_source={"own_cohort", "member_union"}
    )
    assert set(reference_backed_codes) - set(cancer_types.classification_target_codes()) == {"SGC"}
    assert cancer_types.reference_source_codes("member_union") == cancer_types.cancer_type_codes(
        reference_source="member_union"
    )
    assert cancer_types.cancer_type_reference_source("CRC_MSI") == "member_union"
    assert cancer_types.cancer_type_reference_code("CRC_MSI") == "CRC_MSI"
    assert cancer_types.cancer_type_reference_source("STAD_MSI") == "own_cohort"
    assert cancer_types.cancer_type_reference_code("STAD_MSI") == "STAD_MSI"
    assert cancer_types.cancer_type_reference_source("NET_NONPANCREATIC") == "none"
    assert cancer_types.cancer_type_reference_code("NET_NONPANCREATIC") == "NET"
    assert cancer_types.cancer_type_reference_source("SGC") == "member_union"
    assert cancer_types.cancer_type_reference_code("SGC") is None

    assert cancer_types.is_classification_target(None) is False
    assert cancer_types.cancer_type_reference_source(None) is None
    assert cancer_types.cancer_type_reference_code(None) is None
    assert cd.is_classification_target("NET") is True
    assert (
        cd.cancer_ontology.classification_target_codes()
        == cancer_types.classification_target_codes()
    )
    assert cd.cancer_ontology.cancer_type_reference_source("CRC_MSI") == "member_union"
    assert cd.cancer_ontology.cancer_type_reference_code("THYMCA") == "THYM_EPITHELIAL"


def test_cancer_type_siblings_use_parent_hierarchy():
    siblings = cancer_types.cancer_type_siblings("COAD")
    assert siblings["code"].tolist() == ["READ"]
    cross_level = cancer_types.cancer_type_siblings("COAD", same_ontology_level=False)
    assert cross_level["code"].tolist() == ["CRC_MSI", "READ"]
    assert cancer_types.cancer_type_siblings("CRC_MSI").empty
    molecular_siblings = cancer_types.cancer_type_siblings("COAD_MSI")
    assert molecular_siblings["code"].tolist() == ["COAD_MSS"]


def test_cancer_type_records_include_evidence_expression_and_normal_tissue():
    records = cancer_types.cancer_type_records(["COAD_MSI", "CRC_MSI"]).set_index("code")
    assert records.loc["COAD_MSI", "evidence_source_code"] == "CRC_MSI"
    assert records.loc["COAD_MSI", "evidence_source_kind"] == "source_scope"
    assert records.loc["COAD_MSI", "normal_tissue_code"] == "colon"
    assert records.loc["COAD_MSI", "hpa_tissues"] == ("colon",)
    assert bool(records.loc["COAD_MSI", "has_expression_matrix"]) is True

    assert records.loc["CRC_MSI", "evidence_source_code"] == "CRC_MSI"
    assert records.loc["CRC_MSI", "evidence_source_kind"] == "direct"
    assert records.loc["CRC_MSI", "normal_tissue_code"] == "colorectum"
    assert records.loc["CRC_MSI", "hpa_tissues"] == ("colon", "rectum")
    assert bool(records.loc["CRC_MSI", "has_expression_matrix"]) is False


def test_stad_molecular_subtype_rows_have_direct_expression_shards():
    records = cancer_types.cancer_type_records(
        ["STAD", "STAD_MSI", "STAD_EBV", "STAD_CIN", "STAD_GS"]
    ).set_index("code")

    assert records.loc["STAD_MSI", "parent_code"] == "STAD"
    assert records.loc["STAD_MSI", "subtype_groups"] == ("MSI",)
    assert records.loc["STAD_MSI", "normal_tissue_code"] == "stomach"
    assert records.loc["STAD_MSI", "hpa_tissues"] == ("stomach",)
    assert records.loc["STAD_MSI", "source_pmid"] == "PMID:25079317"
    assert bool(records.loc["STAD_MSI", "has_expression_matrix"]) is True
    assert (
        records.loc["STAD_MSI", "source_matrix_cohort"] == "TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE"
    )
    assert records.loc["STAD_MSI", "source_matrix_n_samples"] == 73

    assert records.loc["STAD_EBV", "subtype_groups"] == ("EBV_POS",)
    assert records.loc["STAD_EBV", "subtype_axes"] == ("viral_ebv",)
    assert records.loc["STAD_EBV", "mmr_classifier_role"] == "exclude_confounder"
    assert bool(records.loc["STAD_EBV", "has_expression_matrix"]) is True

    assert records.loc["STAD_CIN", "subtype_groups"] == ("MSS",)
    assert records.loc["STAD_GS", "subtype_groups"] == ("MSS",)
    assert bool(records.loc["STAD_CIN", "has_expression_matrix"]) is True
    assert bool(records.loc["STAD_GS", "has_expression_matrix"]) is True
    assert bool(records.loc["STAD", "has_expression_matrix"]) is True


def test_mmr_status_axis_queries_and_exports():
    positive = cancer_types.mmrd_cancer_codes()
    assert {"CRC_MSI", "COAD_MSI", "READ_MSI", "UCEC_MSI", "STAD_MSI"} <= set(positive)
    assert cancer_types.cancer_mismatch_repair_codes(state="dMMR") == positive
    assert cancer_types.cancer_mismatch_repair_codes(classifier_role="positive") == positive

    assert set(cancer_types.mmrd_cancer_codes(under="CRC")) == {
        "CRC_MSI",
        "COAD_MSI",
        "READ_MSI",
    }
    assert cancer_types.mmrd_cancer_codes(under="STAD") == ["STAD_MSI"]
    assert set(cancer_types.pmmr_cancer_codes(under="UCEC")) == {"UCEC_CNL", "UCEC_CNH"}
    assert cancer_types.pmmr_cancer_codes(under="STAD") == ["STAD_CIN", "STAD_GS"]

    assert set(cancer_types.mmr_confounder_cancer_codes()) >= {"UCEC_POLE", "STAD_EBV"}
    assert cancer_types.mmr_hypermutated_confounder_codes() == ["UCEC_POLE"]

    direct_positive = cancer_types.mmrd_cancer_codes(expression_only=True)
    assert direct_positive == ["COAD_MSI", "READ_MSI", "UCEC_MSI", "STAD_MSI"]
    direct_negative = cancer_types.pmmr_cancer_codes(expression_only=True)
    assert direct_negative == [
        "COAD_MSS",
        "READ_MSS",
        "UCEC_CNL",
        "UCEC_CNH",
        "STAD_CIN",
        "STAD_GS",
    ]

    record = cancer_types.cancer_mismatch_repair_status("STAD_MSI")
    assert record["mmr_axis_state"] == "mmrd"
    assert record["mmr_classifier_role"] == "positive"
    from_records = cancer_types.cancer_mismatch_repair_statuses(
        cancer_types=cancer_types.cancer_type_records(subtype_group="MSI")
    )
    assert from_records["cancer_code"].tolist() == [
        "CRC_MSI",
        "COAD_MSI",
        "READ_MSI",
        "UCEC_MSI",
        "STAD_MSI",
    ]
    assert cancer_types.cancer_mismatch_repair_status(None) is None
    assert cancer_types.cancer_type_records(mmr_state=[]).empty

    ontology = cd.cancer_ontology
    assert ontology.mmrd_cancer_codes(under="STAD") == ["STAD_MSI"]
    assert cd.mmrd_cancer_codes(under="CRC") == ["CRC_MSI", "COAD_MSI", "READ_MSI"]


def test_expression_reference_coverage_contract():
    coverage = cancer_types.expression_reference_coverage(["COAD_MSI", "CRC_MSI", "ASTB"])
    expected = {
        "code",
        "lineage_group",
        "ontology_level",
        "ontology_kind",
        "reference_source",
        "is_classification_target",
        "classification_reference_code",
        "ontology_depth",
        "has_expression_reference",
        "has_direct_expression_reference",
        "has_computed_expression_reference",
        "computed_expression_member_codes",
        "observed_bulk_reference",
        "expression_reference_kind",
        "source_matrix_cohort",
        "source_matrix_n_samples",
        "normalization_method",
        "gene_id_space",
        "data_version",
        "source_matrix_version",
        "has_molecular_definition",
        "molecular_definition_kind",
        "consumer_recommendation",
        "missing_reason",
    }
    assert expected <= set(coverage.columns)

    keyed = coverage.set_index("code")
    assert bool(keyed.loc["COAD_MSI", "has_expression_reference"]) is True
    assert keyed.loc["COAD_MSI", "reference_source"] == "own_cohort"
    assert keyed.loc["COAD_MSI", "classification_reference_code"] == "COAD_MSI"
    assert bool(keyed.loc["COAD_MSI", "has_direct_expression_reference"]) is True
    assert bool(keyed.loc["COAD_MSI", "has_computed_expression_reference"]) is False
    assert keyed.loc["COAD_MSI", "expression_reference_kind"] == "observed_bulk"
    assert keyed.loc["COAD_MSI", "consumer_recommendation"] == "direct_reference"
    assert keyed.loc["COAD_MSI", "normalization_method"] == "clean_tpm_16_9_75"
    assert keyed.loc["COAD_MSI", "gene_id_space"] == "oncoref_canonical_ensg"

    assert bool(keyed.loc["CRC_MSI", "has_direct_expression_reference"]) is False
    assert bool(keyed.loc["CRC_MSI", "has_computed_expression_reference"]) is True
    assert keyed.loc["CRC_MSI", "reference_source"] == "member_union"
    assert keyed.loc["CRC_MSI", "classification_reference_code"] == "CRC_MSI"
    assert keyed.loc["CRC_MSI", "computed_expression_member_codes"] == ("COAD_MSI", "READ_MSI")
    assert keyed.loc["CRC_MSI", "consumer_recommendation"] == "computed_reference"
    assert keyed.loc["CRC_MSI", "molecular_definition_kind"] == ("subtype_group",)
    assert pd.isna(keyed.loc["CRC_MSI", "missing_reason"])

    assert bool(keyed.loc["ASTB", "has_direct_expression_reference"]) is False
    assert keyed.loc["ASTB", "reference_source"] == "none"
    assert keyed.loc["ASTB", "consumer_recommendation"] == "molecular_only"
    assert keyed.loc["ASTB", "molecular_definition_kind"] == ("fusion",)


def test_expression_reference_coverage_computed_groupings():
    coverage = cancer_types.expression_reference_coverage(
        [
            "NET",
            "CRC",
            "CRC_MSI",
            "NSCLC",
            "BTC",
            "SGC",
            "SARC",
            "SARC_LPS",
            "OV",
            "STAD_CIN",
            "STAD_MSI",
            "STAD_GS",
            "STAD_EBV",
            "UCEC_CNH",
            "UCEC_MSI",
            "UCEC_CNL",
            "UCEC_POLE",
        ]
    ).set_index("code")

    for code in ["NET", "CRC", "CRC_MSI", "NSCLC", "BTC", "SARC", "SARC_LPS"]:
        row = coverage.loc[code]
        assert bool(row["has_expression_reference"]) is True
        assert bool(row["has_direct_expression_reference"]) is False
        assert bool(row["has_computed_expression_reference"]) is True
        assert row["expression_reference_kind"] == "computed_union"
        assert row["expression_reference_source_code"] == code
        assert row["reference_source"] == "member_union"
        assert row["classification_reference_code"] == code
        assert row["consumer_recommendation"] == "computed_reference"
        assert pd.isna(row["missing_reason"])
        assert row["computed_expression_member_codes"]

    sgc = coverage.loc["SGC"]
    assert bool(sgc["has_computed_expression_reference"]) is True
    assert sgc["expression_reference_kind"] == "computed_union"
    assert sgc["expression_reference_source_code"] == "SGC"
    assert sgc["reference_source"] == "member_union"
    assert pd.isna(sgc["classification_reference_code"])
    assert bool(sgc["is_classification_target"]) is False
    assert sgc["consumer_recommendation"] == "reference_only"

    assert coverage.loc["CRC_MSI", "computed_expression_member_codes"] == ("COAD_MSI", "READ_MSI")
    assert bool(coverage.loc["OV", "has_direct_expression_reference"]) is True
    assert bool(coverage.loc["OV", "has_computed_expression_reference"]) is False
    assert coverage.loc["OV", "expression_reference_kind"] == "observed_bulk"
    assert coverage.loc["OV", "reference_source"] == "own_cohort"

    for code, n_samples in {
        "STAD_CIN": 221,
        "STAD_MSI": 73,
        "STAD_GS": 50,
        "STAD_EBV": 30,
        "UCEC_CNH": 85,
        "UCEC_MSI": 41,
        "UCEC_CNL": 30,
        "UCEC_POLE": 16,
    }.items():
        assert bool(coverage.loc[code, "has_expression_reference"]) is True
        assert bool(coverage.loc[code, "has_direct_expression_reference"]) is True
        assert bool(coverage.loc[code, "has_computed_expression_reference"]) is False
        assert coverage.loc[code, "expression_reference_kind"] == "observed_bulk"
        assert coverage.loc[code, "expression_reference_source_code"] == code
        assert coverage.loc[code, "source_matrix_n_samples"] == n_samples
        assert coverage.loc[code, "consumer_recommendation"] == "direct_reference"


def test_expression_reference_coverage_filters_and_empty_results():
    crc = cancer_types.expression_reference_coverage(subtype_group="MSI", under="CRC")
    assert crc["code"].tolist() == ["CRC_MSI", "COAD_MSI", "READ_MSI"]
    assert set(crc["consumer_recommendation"]) == {"direct_reference", "computed_reference"}

    empty = cancer_types.expression_reference_coverage([])
    assert empty.empty
    assert list(empty.columns) == list(cancer_types.expression_reference_coverage(["PRAD"]).columns)


def test_coverage_for_cancer_type_single_record_and_exports():
    record = cancer_types.coverage_for_cancer_type("prostate")
    assert record["code"] == "PRAD"
    assert record["consumer_recommendation"] == "direct_reference"
    assert cancer_types.coverage_for_cancer_type(None) is None
    with pytest.raises(ValueError):
        cancer_types.coverage_for_cancer_type("not_a_real_cancer")
    assert cd.coverage_for_cancer_type("PRAD")["code"] == "PRAD"
    assert cd.expression_reference_coverage(["PRAD"]).iloc[0]["code"] == "PRAD"
    assert cd.cancer_ontology.coverage_for_cancer_type("PRAD")["code"] == "PRAD"


def test_btc_records_source_scope_chol_and_gbc():
    records = cancer_types.cancer_type_records(["BTC", "CHOL", "GBC"]).set_index("code")
    assert records.loc["BTC", "evidence_source_code"] == "BTC"
    assert records.loc["BTC", "evidence_source_kind"] == "direct"
    assert records.loc["BTC", "children"] == ("CHOL", "GBC")
    assert records.loc["BTC", "normal_tissue_code"] == "gallbladder"

    assert records.loc["CHOL", "parent_code"] == "BTC"
    assert records.loc["CHOL", "evidence_source_code"] == "BTC"
    assert records.loc["CHOL", "evidence_source_kind"] == "source_scope"
    assert records.loc["GBC", "parent_code"] == "BTC"
    assert records.loc["GBC", "evidence_source_code"] == "BTC"
    assert records.loc["GBC", "evidence_source_kind"] == "source_scope"


def test_sgc_records_source_scope_salivary_children():
    assert cancer_types.resolve_cancer_type("salivary gland") == "SGC"
    records = cancer_types.cancer_type_records(["SGC", "ACINIC", "ADCC"]).set_index("code")
    registry = cancer_types.cancer_type_registry().set_index("code")
    info = cancer_types.cancer_type_info("SGC")
    assert records.loc["SGC", "evidence_source_code"] == "SGC"
    assert records.loc["SGC", "evidence_source_kind"] == "direct"
    assert records.loc["SGC", "children"] == ("ACINIC", "ADCC")
    assert records.loc["SGC", "normal_tissue_code"] == "salivary_gland"
    assert records.loc["SGC", "hpa_tissues"] == ("salivary gland",)
    assert records.loc["SGC", "reference_source"] == "member_union"
    assert bool(records.loc["SGC", "is_classification_target"]) is False
    assert pd.isna(records.loc["SGC", "classification_reference_code"])
    assert bool(registry.loc["SGC", "is_classification_target"]) is False
    assert info["is_classification_target"] is False
    assert info["classification_reference_code"] is None

    assert records.loc["ACINIC", "parent_code"] == "SGC"
    assert records.loc["ACINIC", "evidence_source_code"] == "SGC"
    assert records.loc["ACINIC", "evidence_source_kind"] == "source_scope"
    assert records.loc["ADCC", "parent_code"] == "SGC"
    assert records.loc["ADCC", "evidence_source_code"] == "SGC"
    assert records.loc["ADCC", "evidence_source_kind"] == "source_scope"


def test_net_nonpancreatic_records_source_scope_site_codes():
    assert cancer_types.resolve_cancer_type("nonpancreatic NET") == "NET_NONPANCREATIC"
    records = cancer_types.cancer_type_records(
        ["NET_NONPANCREATIC", "NET_LUNG", "NET_MIDGUT", "NET_RECTAL", "NET_PANCREAS"]
    ).set_index("code")
    assert records.loc["NET_NONPANCREATIC", "parent_code"] == "NET"
    assert records.loc["NET_NONPANCREATIC", "evidence_source_code"] == "NET_NONPANCREATIC"
    assert records.loc["NET_NONPANCREATIC", "evidence_source_kind"] == "direct"
    assert records.loc["NET_NONPANCREATIC", "normal_tissue_code"] == "neuroendocrine"

    for code in ("NET_LUNG", "NET_MIDGUT", "NET_RECTAL"):
        assert records.loc[code, "parent_code"] == "NET"
        assert records.loc[code, "evidence_source_code"] == "NET_NONPANCREATIC"
        assert records.loc[code, "evidence_source_kind"] == "source_scope"

    assert records.loc["NET_PANCREAS", "evidence_source_code"] == "NET_PANCREAS"
    assert records.loc["NET_PANCREAS", "evidence_source_kind"] == "direct"


def test_extrapulmonary_high_grade_nen_is_context_aggregate_not_lung_lcnec():
    assert cancer_types.resolve_cancer_type("extrapulmonary G3 NEN") == ("NEN_EXTRAPULMONARY_HG")
    assert cancer_types.resolve_cancer_type("NEN_G3_EXTRAPULMONARY") == ("NEN_EXTRAPULMONARY_HG")
    records = cancer_types.cancer_type_records(
        ["NEN_EXTRAPULMONARY_HG", "NEC_LUNG_LARGECELL"]
    ).set_index("code")
    assert cancer_types.is_mixture_cohort("NEN_EXTRAPULMONARY_HG") is True
    assert records.loc["NEN_EXTRAPULMONARY_HG", "parent_code"] == "NEN"
    assert records.loc["NEN_EXTRAPULMONARY_HG", "primary_tissue"] == "neuroendocrine"
    assert records.loc["NEN_EXTRAPULMONARY_HG", "differentiation"] == "NEN_G3"
    assert records.loc["NEN_EXTRAPULMONARY_HG", "grade_tier"] == "high"
    assert records.loc["NEN_EXTRAPULMONARY_HG", "evidence_source_code"] == ("NEN_EXTRAPULMONARY_HG")
    assert records.loc["NEC_LUNG_LARGECELL", "evidence_source_code"] == ("NEC_LUNG_LARGECELL")
    assert "NEN_EXTRAPULMONARY_HG" in cancer_types.cancer_type_codes(grade_tier="high")
    assert "NEN_EXTRAPULMONARY_HG" in cancer_types.cancer_type_codes(differentiation="NEN_G3")


def test_normal_tissue_map_uses_hpa_rna_v23_tissue_names():
    tissue_map = cancer_types.cancer_normal_tissue_map().set_index("primary_tissue")
    used = {
        str(tissue).lower() for tissues in tissue_map["hpa_tissues"] for tissue in (tissues or ())
    }
    assert used <= HPA_RNA_V23_TISSUES
    assert tissue_map.loc["soft_tissue", "hpa_tissues"] == ()
    assert tissue_map.loc["soft_tissue", "match_confidence"] == "unresolved"
    assert tissue_map.loc["oral_cavity", "hpa_tissues"] == ("tongue",)
    assert tissue_map.loc["pharynx", "hpa_tissues"] == ("tonsil",)
    assert tissue_map.loc["biliary_tract", "hpa_tissues"] == ("gallbladder",)
    assert tissue_map.loc["salivary_gland", "hpa_tissues"] == ("salivary gland",)


def test_cancer_type_reference_data_joins_scalar_oncoref_metrics():
    refs = cancer_types.cancer_type_reference_data(["COAD_MSI", "READ_MSI"]).set_index("code")
    assert refs.loc["COAD_MSI", "evidence_source_code"] == "CRC_MSI"
    assert refs.loc["COAD_MSI", "ici_response_source_code"] == "CRC_MSI"
    assert refs.loc["COAD_MSI", "ici_inheritance_kind"] == "source_scope"
    assert refs.loc["COAD_MSI", "tmb"] == 46.0
    assert refs.loc["READ_MSI", "normal_tissue_code"] == "rectum"
    assert bool(refs.loc["READ_MSI", "has_expression_matrix"]) is True
    assert refs.loc["READ_MSI", "us_incidence_pct"] is not None


def test_cancer_type_reference_data_honors_inherit_false_for_tmb():
    refs = cancer_types.cancer_type_reference_data(["LUAD_KRAS"], inherit=False).set_index("code")
    assert refs.loc["LUAD_KRAS", "tmb"] is None


def test_matched_normal_tissue_expression_uses_hpa_tissue_match(monkeypatch):
    import pandas as pd

    from oncoref import hpa

    hpa_fixture = pd.DataFrame(
        {
            "Gene": ["ENSG1", "ENSG1", "ENSG2"],
            "Gene name": ["G1", "G1", "G2"],
            "Tissue": ["colon", "rectum", "colon"],
            "nTPM": [10.0, 20.0, 30.0],
        }
    )
    monkeypatch.setattr(hpa, "hpa_rna_consensus", lambda: hpa_fixture.copy())
    out = cancer_types.matched_normal_tissue_expression("COAD", genes="G1")
    assert out["cancer_code"].unique().tolist() == ["COAD"]
    assert out["normal_tissue_code"].unique().tolist() == ["colon"]
    assert out["Tissue"].tolist() == ["colon"]
    assert out["nTPM"].tolist() == [10.0]


def test_matched_normal_tissue_missing_input_is_unresolved():
    assert cancer_types.matched_normal_tissue(None) is None
    assert cancer_types.matched_normal_tissue("not_a_real_cancer") is None
    assert cancer_types.matched_normal_tissue_expression(None).empty
    assert cancer_types.matched_normal_tissue_expression("not_a_real_cancer").empty


def test_matched_normal_tissue_expression_uses_hpa_rna_consensus_names(monkeypatch):
    import pandas as pd

    from oncoref import hpa

    hpa_fixture = pd.DataFrame(
        {
            "Gene": ["ENSG1", "ENSG1", "ENSG1", "ENSG1", "ENSG1", "ENSG1"],
            "Gene name": ["G1", "G1", "G1", "G1", "G1", "G1"],
            "Tissue": ["cervix", "endometrium", "skin", "stomach", "tongue", "tonsil"],
            "nTPM": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    monkeypatch.setattr(hpa, "hpa_rna_consensus", lambda: hpa_fixture.copy())

    assert cancer_types.matched_normal_tissue_expression("CESC", genes="G1")["Tissue"].tolist() == [
        "cervix"
    ]
    assert cancer_types.matched_normal_tissue_expression("UCEC", genes="G1")["Tissue"].tolist() == [
        "endometrium"
    ]
    assert cancer_types.matched_normal_tissue_expression("SKCM", genes="G1")["Tissue"].tolist() == [
        "skin"
    ]
    assert cancer_types.matched_normal_tissue_expression("STAD", genes="G1")["Tissue"].tolist() == [
        "stomach"
    ]
    assert cancer_types.matched_normal_tissue_expression("HNSC_HPVneg", genes="G1")[
        "Tissue"
    ].tolist() == ["tongue"]
    assert cancer_types.matched_normal_tissue_expression("HNSC", genes="G1")["Tissue"].tolist() == [
        "tonsil"
    ]
    assert cancer_types.matched_normal_tissue_expression("SARC", genes="G1").empty
    assert cancer_types.matched_normal_tissue_expression("NPC", genes="G1").empty
