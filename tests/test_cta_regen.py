# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CTA regeneration from HPA — parity with the shipped table (#35, Phase C)."""

import io
import os
from pathlib import Path

import pandas as pd
import pytest

from oncoref import cta_regen, reference_data

_CSV = Path(__file__).resolve().parents[1] / "oncoref" / "data" / "cancer-testis-antigens.csv"

_HPA_READY = reference_data.is_cached("hpa_rna_consensus") and reference_data.is_cached(
    "hpa_normal_tissue"
)


def _require_hpa_v23():
    if _HPA_READY:
        return
    if os.environ.get("CI"):
        pytest.fail("CI must stage the pinned HPA v23 RNA and normal-tissue tables")
    pytest.skip("HPA v23 not cached")


def test_fraction_deflation_semantics():
    # All-below-1 nTPM -> deflated total 0 -> 1.0 (restricted by the +1 pseudocount).
    f = cta_regen._fraction
    repro = frozenset({"testis"})
    assert f({"testis": 0.5, "liver": 0.4}, repro, deflate=True) == 1.0
    assert f({"testis": 0.5, "liver": 0.4}, repro, deflate=False) < 1.0
    # A real somatic signal lowers the deflated reproductive fraction.
    val = f({"testis": 10.0, "liver": 10.0}, repro, deflate=True)
    assert 0.0 < val < 1.0


def test_regeneration_reproduces_shipped_table():
    _require_hpa_v23()

    # Regenerating the 47 HPA columns from HPA v23 must reproduce the shipped,
    # oncoref-owned table exactly — proof the regenerator is the source of truth.
    regen = cta_regen.regenerate_cta_columns(pd.read_csv(_CSV))
    buf = io.StringIO()
    regen.to_csv(buf, index=False)
    from_regen = pd.read_csv(io.StringIO(buf.getvalue()), dtype=str, keep_default_na=False)
    shipped = pd.read_csv(_CSV, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(from_regen, shipped)


def test_regeneration_preserves_identity_columns():
    _require_hpa_v23()

    old = pd.read_csv(_CSV)
    regen = cta_regen.regenerate_cta_columns(old)
    for col in ("Symbol", "Ensembl_Gene_ID", "Aliases", "source_databases", "biotype"):
        pd.testing.assert_series_equal(regen[col], old[col], check_dtype=False)


def test_rna_only_below_floor_confidence_is_capped():
    # tsarina#114 cap: no gene with no protein and rna_max_ntpm < 2.0 may keep a
    # HIGH restriction_confidence (near-noise RNA must not earn HIGH).
    df = pd.read_csv(_CSV, dtype=str, keep_default_na=False)
    rna_max = pd.to_numeric(df["rna_max_ntpm"], errors="coerce")
    offenders = df[
        (df["restriction_confidence"] == "HIGH")
        & (df["protein_restriction"] == "NO_DATA")
        & (rna_max < 2.0)
    ]
    assert offenders.empty, sorted(offenders["Symbol"])
