# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import hashlib
import re

import pandas as pd
import pytest

from oncoref import incidence
from oncoref import load_dataset as ld


def test_derived_burden_cache_invalidates_on_clear(monkeypatch):
    # The lru_cached burden maps must be cleared by load_dataset._clear_cache(),
    # so swapping a bundled fixture is reflected (test-isolation guard).
    incidence._tissue_burden_map()  # populate the cache
    real = incidence.get_data

    def fake(name, *a, **k):
        if name == "tissue-burden-map":
            return pd.DataFrame(
                {"primary_tissue": ["unobtanium"], "burden_category": ["xyzzy"], "scope": ["solid"]}
            )
        return real(name, *a, **k)

    monkeypatch.setattr(incidence, "get_data", fake)
    ld._clear_cache()
    try:
        assert incidence._tissue_burden_map() == {"unobtanium": "xyzzy"}
    finally:
        monkeypatch.undo()
        ld._clear_cache()  # restore real data for other tests


def test_burden_map_nonempty_floats():
    mapping = incidence.cancer_burden()
    assert mapping
    assert all(isinstance(v, float) for v in mapping.values())


def test_burden_metric_validation():
    with pytest.raises(ValueError):
        incidence.cancer_burden(metric="not_a_metric")


def test_burden_all_metrics_resolve():
    for metric in incidence._BURDEN_METRICS:
        mapping = incidence.cancer_burden(metric=metric)
        assert mapping


def test_burden_table_exposes_source_provenance_schema():
    df = incidence.cancer_burden_df()
    expected = {
        "us_incidence_count",
        "us_incidence_total",
        "us_mortality_count",
        "us_mortality_total",
        "world_incidence_count",
        "world_incidence_total",
        "world_mortality_count",
        "world_mortality_total",
        "us_source_locator",
        "us_source_locator_status",
        "world_source_locator",
        "world_source_locator_status",
        "source_site_labels",
        "source_site_codes",
        "included_source_sites",
        "excluded_source_sites",
        "derivation_basis",
        "rounding_rule",
        "provenance_notes",
        "aggregation",
        "source_anchor",
    }
    assert expected <= set(df.columns)
    assert set(df["us_source_locator_status"]) == {"not_extracted"}
    assert set(df["world_source_locator_status"]) == {"not_extracted"}
    assert set(df["rounding_rule"]) == {"not_extracted"}
    assert set(df["derivation_basis"]) <= {
        "direct_source",
        "not_extracted",
        "sum_of_sites",
        "residual",
        "literature_approximation",
    }
    by_category = df.set_index("burden_category")
    assert by_category.loc["colorectal", "derivation_basis"] == "sum_of_sites"
    assert by_category.loc["other_and_unknown_primary", "derivation_basis"] == "residual"


def test_burden_provenance_is_a_lossless_pirlygenes_replacement():
    df = incidence.cancer_burden_df()
    assert not df["aggregation"].isna().any()
    assert not df["source_anchor"].isna().any()
    assert df["aggregation"].astype(str).str.strip().ne("").all()
    assert df["source_anchor"].astype(str).str.strip().ne("").all()

    # Frozen from pirlygenes' 37 populated burden-category aggregation values.
    ordered = df.sort_values("burden_category")
    aggregation_payload = "\n".join(
        f"{row.burden_category}\t{row.aggregation}" for row in ordered.itertuples()
    )
    assert hashlib.sha256(aggregation_payload.encode()).hexdigest() == (
        "0a1e3af0cf10da3fb43e180ca0d0c6d0f7cd081f53af3a4c58f3bcab48cbd589"
    )

    anchor_pattern = re.compile(r"(?:PMID:\d+|DOI:10\.\d{4,9}/\S+)")
    for source_anchor in df["source_anchor"].astype(str):
        assert all(anchor_pattern.fullmatch(token) for token in source_anchor.split(";"))

    by_category = df.set_index("burden_category")
    assert by_category.loc["adrenal", "source_anchor"] == "PMID:29977421;PMID:33112261"
    assert by_category.loc["soft_tissue_sarcoma", "source_anchor"] == (
        "PMID:38230766;PMID:39498209"
    )
    assert by_category.loc["kaposi_sarcoma", "source_anchor"] == (
        "DOI:10.3322/caac.21834;DOI:10.1016/S2214-109X(23)00349-2"
    )

    acs_rows = df["source"].astype(str).str.contains("ACS-CFF2024")
    globocan_rows = df["source"].astype(str).str.contains("GLOBOCAN2022")
    assert df.loc[acs_rows, "source_anchor"].str.contains("PMID:38230766", regex=False).all()
    assert (
        df.loc[globocan_rows, "source_anchor"]
        .str.contains("DOI:10.3322/caac.21834", regex=False)
        .all()
    )

    head_and_neck = by_category.loc["head_and_neck"]
    assert head_and_neck["us_mortality_pct"] == 2.6
    assert "downstream 1.8% omitted larynx" in head_and_neck["provenance_notes"]


def test_burden_category_from_primary_tissue():
    assert incidence.burden_category("PRAD") == "prostate"
    assert incidence.burden_category("LUAD") == "lung"


def test_burden_category_sarcoma_bone_vs_soft():
    # Sarcoma family splits on primary_tissue: bone vs soft tissue.
    assert incidence.burden_category("SARC_OS") == "bone_and_joint"
    assert incidence.burden_category("SARC") == "soft_tissue_sarcoma"


def test_burden_category_override_table():
    # cancer-code-burden-map.csv carries the true ontology exceptions.
    overrides = incidence.cancer_code_burden_map()
    assert overrides
    for code, expected in overrides.items():
        assert incidence.burden_category(code) == expected


def test_burden_category_unknown_returns_none():
    assert incidence.burden_category("not_a_real_cancer") is None


def test_every_registry_code_resolves_to_a_burden_category():
    # Drift guard: the primary_tissue/family -> burden maps in incidence.py must
    # cover the registry's vocabulary. A new registry primary_tissue/family that
    # nothing maps would silently yield None here.
    from oncoref.cancer_types import cancer_type_registry

    registry = cancer_type_registry()
    unmapped = [
        code for code in registry["code"].astype(str) if incidence.burden_category(code) is None
    ]
    assert not unmapped, f"registry codes with no burden category: {unmapped}"


def test_family_burden_map_has_no_stale_families():
    # Drift guard for the data-driven family fallback: every family key must be a
    # current registry family. (This caught the stale 'cns' slug left after the
    # registry split cns -> cns-glial/-embryonal/... .)
    from oncoref.cancer_types import cancer_type_registry

    registry_families = set(cancer_type_registry()["family"].dropna().astype(str))
    stale = [f for f in incidence._family_burden_map() if f not in registry_families]
    assert not stale, f"family-burden-map keys not in the registry: {stale}"
