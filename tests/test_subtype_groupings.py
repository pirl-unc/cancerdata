# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Cross-cutting subtype groupings: MSI/POLE/HPV reverse queries (#27, O2)."""

from oncoref import (
    cancer_subtype_group,
    cancer_subtype_groupings,
    cancer_type_registry,
)


def test_all_msi_types():
    assert cancer_subtype_group("MSI") == [
        "CRC_MSI",
        "COAD_MSI",
        "READ_MSI",
        "UCEC_MSI",
        "STAD_MSI",
    ]


def test_all_pole_subtypes():
    assert cancer_subtype_group("POLE") == ["UCEC_POLE"]


def test_infantile_mapk_spindle_is_a_biologic_relationship_not_a_synonym():
    assert cancer_subtype_group("INFANTILE_MAPK_SPINDLE") == ["SARC_IFS", "CMN"]


def test_under_scopes_to_descendants():
    # MSI across all cancers vs colorectal-only (descendants of CRC).
    assert cancer_subtype_group("MSI", under="CRC") == ["CRC_MSI", "COAD_MSI", "READ_MSI"]
    assert cancer_subtype_group("MSI", under="UCEC") == ["UCEC_MSI"]
    assert cancer_subtype_group("MSI", under="STAD") == ["STAD_MSI"]
    assert cancer_subtype_group("MSS", under="STAD") == ["STAD_CIN", "STAD_GS"]


def test_unknown_group_is_empty():
    assert cancer_subtype_group("NOT_A_GROUP") == []


def test_groupings_table_axes():
    df = cancer_subtype_groupings()
    assert set(df.columns) == {"group_code", "axis", "member_code", "basis"}
    assert {"microsatellite", "hypermutation", "viral_hpv", "viral_ebv"} <= set(df["axis"])


def test_every_member_is_a_registry_code():
    # A grouping must never reference a cancer code that doesn't exist, or the
    # reverse query would return phantom codes.
    members = set(cancer_subtype_groupings()["member_code"])
    codes = set(cancer_type_registry()["code"])
    missing = sorted(members - codes)
    assert not missing, f"subtype-grouping members not in the registry: {missing}"
