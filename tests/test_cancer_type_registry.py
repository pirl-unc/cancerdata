# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Integrity guards for the oncoref-owned cancer-type registry.

oncoref owns the cancer-type ontology outright — the registry is authoritative
here, not mirrored from anywhere. These guards pin the schema and the structural
invariants the navigation/grouping/fusion accessors rely on.
"""

from pathlib import Path

import pandas as pd

from oncoref import (
    cancer_type_records,
    cancer_type_registry,
    cancer_type_subtypes_of,
    cancer_types,
    cohort_aggregate_members,
    cohort_registry_df,
    resolve_cancer_type,
)

_CSV = Path(__file__).resolve().parents[1] / "oncoref" / "data" / "cancer-type-registry.csv"

# The registry schema (shipped column order).
REGISTRY_COLUMNS = [
    "code",
    "name",
    "family",
    "primary_tissue",
    "primary_template",
    "parent_code",
    "ontology_level",
    "ontology_kind",
    "is_classification_target",
    "subtype_key",
    "expression_source",
    "source_cohort",
    "source_pmid",
    "notes",
    "mixture_cohort",
    "pediatric",
    "differentiation",
    "grade_tier",
    "viral_etiology",
    "viral_agent",
    "fusion_driven",
    "fusion_driver",
]


def test_schema_matches_contract():
    assert list(pd.read_csv(_CSV, nrows=0).columns) == REGISTRY_COLUMNS


def test_no_duplicate_codes():
    codes = pd.read_csv(_CSV, dtype=str)["code"]
    dups = sorted(codes[codes.duplicated()])
    assert not dups, f"duplicate cancer-type codes: {dups}"


def test_parent_code_referential_integrity():
    # Every parent_code must name an existing code — no orphan branches in the tree.
    df = pd.read_csv(_CSV, dtype=str, keep_default_na=False)
    codes = set(df["code"])
    orphans = sorted({p for p in df["parent_code"] if p and p not in codes})
    assert not orphans, f"parent_code(s) with no matching code: {orphans}"


def test_crc_hierarchy():
    assert set(cancer_type_subtypes_of("CRC")) >= {"COAD", "READ"}


def test_cmn_is_canonical_and_ifs_remains_a_classification_target():
    raw = pd.read_csv(_CSV, dtype=str, keep_default_na=False).set_index("code")
    public = cancer_types.cancer_type_registry().set_index("code")

    assert resolve_cancer_type("cmn") == "CMN"
    assert resolve_cancer_type("congenital_mesoblastic_nephroma") == "CMN"
    assert resolve_cancer_type("CMN") not in {"SARC_IFS", "WILMS", "SARC"}
    assert raw.loc["CMN", "parent_code"] == ""
    assert raw.loc["CMN", "is_classification_target"] == "False"
    assert bool(public.loc["CMN", "is_classification_target"]) is False
    assert raw.loc["SARC_IFS", "is_classification_target"] == "True"
    assert bool(public.loc["SARC_IFS", "is_classification_target"]) is True


def test_computed_expression_sources_have_members():
    df = pd.read_csv(_CSV, dtype=str, keep_default_na=False)
    cohorts = cohort_registry_df().set_index("cohort_id")
    bad: list[str] = []
    for row in df[df["expression_source"].str.lower() == "computed"].to_dict("records"):
        code = row["code"]
        source_cohort = row["source_cohort"]
        direct_members = cohort_aggregate_members(code)
        source_members = ()
        if source_cohort and source_cohort in cohorts.index:
            source_members = tuple(
                m for m in str(cohorts.loc[source_cohort, "member_cohorts"]).split(";") if m
            )
        if not direct_members and not source_members:
            bad.append(code)
    assert not bad, f"computed registry rows with no aggregate members: {bad}"


def test_source_scoped_clinical_aggregates_are_not_expression_computed():
    records = cancer_type_records(["CRC_MSI", "BTC", "NSCLC", "SGC"]).set_index("code")
    assert records.loc["CRC_MSI", "expression_source"] == "curated"
    assert records.loc["CRC_MSI", "source_cohort"] == "LITERATURE_CURATED"
    assert bool(records.loc["CRC_MSI", "has_expression_matrix"]) is False
    assert records.loc["CRC_MSI", "reference_source"] == "member_union"
    assert records.loc["CRC_MSI", "classification_reference_code"] == "CRC_MSI"
    assert cohort_aggregate_members("CRC_MSI") == ["COAD_MSI", "READ_MSI"]

    assert records.loc["NSCLC", "expression_source"] == "curated"
    assert records.loc["NSCLC", "source_cohort"] == "LITERATURE_CURATED"
    assert bool(records.loc["NSCLC", "has_expression_matrix"]) is False
    assert records.loc["NSCLC", "reference_source"] == "member_union"
    assert cohort_aggregate_members("BTC") == ["CHOL", "GBC"]
    assert records.loc["BTC", "reference_source"] == "none"
    assert bool(records.loc["BTC", "is_classification_target"]) is False
    assert cohort_aggregate_members("SGC") == ["ACINIC", "ADCC"]
    assert cohort_aggregate_members("NSCLC") == ["LUAD", "LUSC"]


def test_source_scope_mixture_targets_follow_reviewed_registry_policy():
    raw = pd.read_csv(_CSV, dtype=str, keep_default_na=False).set_index("code")
    records = cancer_type_records().set_index("code")
    source_scope_mixtures = raw[
        (raw["ontology_level"].str.lower() == "grouping")
        & (raw["ontology_kind"].str.lower() == "source_scope")
        & raw["parent_code"].eq("")
        & raw["mixture_cohort"].str.lower().isin({"true", "1", "yes"})
    ]

    declared_targets = set(
        source_scope_mixtures.index[
            source_scope_mixtures["is_classification_target"].str.lower().isin({"true", "1", "yes"})
        ]
    )
    effective_targets = set(
        source_scope_mixtures.index[
            records.loc[source_scope_mixtures.index, "is_classification_target"].to_numpy(
                dtype=bool
            )
        ]
    )
    assert declared_targets == {"BTC", "NSCLC"}
    assert effective_targets == {"NSCLC"}
    assert set(source_scope_mixtures.index) == {"BTC", "NSCLC", "SGC"}
    assert records.loc[source_scope_mixtures.index, "reference_source"].to_dict() == {
        "BTC": "none",
        "NSCLC": "member_union",
        "SGC": "member_union",
    }


def test_nec_merkel_registry_points_to_built_expression_source():
    records = cancer_type_records(["NEC_MERKEL"]).set_index("code")
    assert records.loc["NEC_MERKEL", "expression_source"] == "GEO"
    assert records.loc["NEC_MERKEL", "source_cohort"] == "GSE235092_MERKEL_2024"
    assert records.loc["NEC_MERKEL", "source_matrix_cohort"] == "GSE235092_MERKEL_2024"
    assert records.loc["NEC_MERKEL", "source_matrix_n_samples"] == 91
    assert bool(records.loc["NEC_MERKEL", "has_expression_matrix"]) is True


def test_registry_has_expected_scale():
    # Sanity floor so an accidental truncation is caught.
    assert len(cancer_type_registry()) >= 159
