# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import pandas as pd
import pytest

from oncoref import coverage

# A controlled 3-gene × 4-patient fixture (threshold 5):
#   gA expressed in p0,p1   gB in p2   gC in p1 (redundant with gA)   p3 uncovered
_GENES = ["ENSG_A", "ENSG_B", "ENSG_C"]
_FIXTURE = pd.DataFrame(
    {
        "Ensembl_Gene_ID": _GENES,
        "Symbol": ["GA", "GB", "GC"],
        "p0": [10.0, 0.0, 0.0],
        "p1": [10.0, 0.0, 10.0],
        "p2": [0.0, 10.0, 0.0],
        "p3": [0.0, 0.0, 0.0],
    }
)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(coverage, "per_sample_expression", lambda *a, **k: _FIXTURE.copy())
    return _GENES


def test_patient_fractions(patched):
    pf = coverage.cta_patient_fractions("X", threshold_tpm=5, gene_ids=patched)
    frac = dict(zip(pf["Ensembl_Gene_ID"], pf["fraction_expressing"]))
    assert frac["ENSG_A"] == 0.5  # p0, p1
    assert frac["ENSG_B"] == 0.25  # p2
    assert frac["ENSG_C"] == 0.25  # p1
    assert (pf["n_patients"] == 4).all()
    # sorted by prevalence descending
    assert next(iter(pf["Ensembl_Gene_ID"])) == "ENSG_A"


def test_addressable_fraction_is_the_union(patched):
    # Covered patients = p0,p1,p2 (p3 expresses nothing) -> 3/4. NOT the 0.5+0.25+0.25
    # sum of per-gene fractions.
    af = coverage.addressable_fraction("X", threshold_tpm=5, gene_ids=patched)
    assert af == 0.75
    # The union is >= the best single gene and <= the naive sum.
    assert af >= 0.5
    assert af < 0.5 + 0.25 + 0.25


def test_greedy_coverage_is_set_cover(patched):
    gc = coverage.greedy_coverage("X", threshold_tpm=5, gene_ids=patched)
    # gA (covers p0,p1) first, then gB (covers p2); gC adds nothing -> excluded.
    assert list(gc["Symbol"]) == ["GA", "GB"]
    assert list(gc["marginal_patients"]) == [2, 1]
    assert gc["cumulative_fraction"].iloc[-1] == 0.75  # == addressable_fraction
    # cumulative is monotonic non-decreasing
    assert (gc["cumulative_fraction"].diff().dropna() > 0).all()

    gene_level = coverage.greedy_coverage("X", threshold_tpm=5, gene_ids=patched, proteoform=False)
    assert "proteoform_key" not in gene_level.columns


def test_mean_antigens_per_patient(patched):
    # hits: gA in p0,p1 (2) + gB in p2 (1) + gC in p1 (1) = 4 over 4 patients -> 1.0.
    # Equals the sum of per-gene prevalences (0.5 + 0.25 + 0.25).
    load = coverage.mean_antigens_per_patient("X", threshold_tpm=5, gene_ids=patched)
    assert load == 1.0
    pf = coverage.cta_patient_fractions("X", threshold_tpm=5, gene_ids=patched)
    assert load == pytest.approx(pf["fraction_expressing"].sum())


def test_mean_antigens_per_patient_empty_panel(patched):
    assert coverage.mean_antigens_per_patient("X", gene_ids=[]) == 0.0


def test_resolve_gene_set_from_file(tmp_path):
    panel = tmp_path / "panel.csv"
    panel.write_text("Ensembl_Gene_ID\nENSG_A\nENSG_B.1\n")
    label, ids = coverage.resolve_gene_set(str(panel))
    assert label == "panel"
    assert ids == {"ENSG_A", "ENSG_B"}


def test_patient_coverage_counts_gene_set_file(patched, tmp_path):
    panel = tmp_path / "panel.csv"
    panel.write_text("Ensembl_Gene_ID\nENSG_A\nENSG_B\nENSG_C\n")
    counts = coverage.patient_coverage(str(panel), cohorts=["X"], thresholds=(5,), proteoform=False)
    got = {row.Ensembl_Gene_ID: (row.n_gt5, row.pct_gt5) for row in counts.itertuples()}
    assert got == {
        "ENSG_A": (2, 50.0),
        "ENSG_B": (1, 25.0),
        "ENSG_C": (1, 25.0),
    }
    assert set(counts["cancer_code"]) == {"X"}


def test_greedy_respects_max_genes(patched):
    gc = coverage.greedy_coverage("X", threshold_tpm=5, gene_ids=patched, max_genes=1)
    assert len(gc) == 1
    assert gc["cumulative_fraction"].iloc[0] == 0.5


def test_empty_panel_is_zero(monkeypatch):
    monkeypatch.setattr(coverage, "per_sample_expression", lambda *a, **k: _FIXTURE.copy())
    assert coverage.addressable_fraction("X", gene_ids=[]) == 0.0
    assert coverage.greedy_coverage("X", gene_ids=[]).empty


def test_high_threshold_covers_nobody(patched):
    # Nothing clears 1000 TPM -> 0 addressable, empty greedy panel.
    assert coverage.addressable_fraction("X", threshold_tpm=1000, gene_ids=patched) == 0.0
    assert coverage.greedy_coverage("X", threshold_tpm=1000, gene_ids=patched).empty


def test_proteoform_paralogs_are_summed(monkeypatch):
    # Two identical-protein paralogs (gA1/gA2) each below threshold in a patient but
    # summing above it: proteoform=True must collapse them to one antigen and catch
    # the patient; proteoform=False keeps them split and misses it.
    fixture = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG_A1", "ENSG_A2", "ENSG_B"],
            "Symbol": ["A1", "A2", "B"],
            "p0": [6.0, 6.0, 0.0],  # A1+A2 = 12 (>10); each alone 6 (<10)
            "p1": [0.0, 0.0, 20.0],
        }
    )
    monkeypatch.setattr(coverage, "per_sample_expression", lambda *a, **k: fixture.copy())
    monkeypatch.setattr(coverage, "_panel_ids", lambda gene_ids: {"ENSG_A1", "ENSG_A2", "ENSG_B"})
    # _hit_matrix lazily does `from .proteoforms import proteoform_group_map`, so patch
    # the source symbol it re-resolves on each call.
    import oncoref.proteoforms as pmod

    monkeypatch.setattr(
        pmod, "proteoform_group_map", lambda *, scope="cta": {"A1/A2": ["ENSG_A1", "ENSG_A2"]}
    )

    pf_sum = coverage.cta_patient_fractions("X", threshold_tpm=10, proteoform=True)
    # A1/A2 collapsed to one row keyed by the contracted symbol "A1/2" (members
    # "A1/A2" in provenance), expressed in p0 (summed 12 > 10) -> fraction 0.5
    a_row = pf_sum[pf_sum["Symbol"] == "A1/2"]
    assert len(a_row) == 1 and a_row["fraction_expressing"].iloc[0] == 0.5
    assert a_row["proteoform_members"].iloc[0] == "A1/A2"
    # per-gene view: neither A1 nor A2 clears 10 alone -> 0 in p0
    pf_split = coverage.cta_patient_fractions("X", threshold_tpm=10, proteoform=False)
    assert pf_split[pf_split["Symbol"] == "A1"]["fraction_expressing"].iloc[0] == 0.0


def test_within_sample_percentile_coverage_preserves_patient_cooccurrence(monkeypatch):
    # Ten biological rows make p90 and p95 easy to inspect: each CTA is the top
    # transcript in a different patient, so the true CTA union covers both patients.
    biological = pd.DataFrame(
        {
            "proteoform_key": ["CTA_A", "CTA_B", *[f"BG{i}" for i in range(8)]],
            "Ensembl_Gene_ID": ["CTA_A", "CTA_B", *[f"BG{i}" for i in range(8)]],
            "Symbol": ["CTA-A", "CTA-B", *[f"BG{i}" for i in range(8)]],
            "proteoform_members": ["CTA-A", "CTA-B", *[f"BG{i}" for i in range(8)]],
            "patient_1": [100.0, 0.0, *range(1, 9)],
            "patient_2": [0.0, 100.0, *range(1, 9)],
        }
    )
    calls = []

    def fake_biological(code, **kwargs):
        calls.append((code, kwargs))
        return biological.copy()

    import oncoref.expression as expression

    monkeypatch.setattr(expression, "_biological_per_sample", fake_biological)
    monkeypatch.setattr(coverage, "_panel_ids", lambda gene_ids: {"CTA_A", "CTA_B"})
    import oncoref.proteoforms as proteoforms

    monkeypatch.setattr(
        proteoforms,
        "gene_to_proteoform_id",
        lambda ids, scope="cta": {gene_id: gene_id for gene_id in ids},
    )

    sweep = coverage.within_sample_percentile_coverage_sweep(
        ["NOT_A_REAL_CODE"], percentiles=(0.90, 0.95)
    )

    assert len(calls) == 1
    assert calls[0][1] == {
        "proteoform": True,
        "auto_fetch": False,
        "scope": "cta",
        "sample_qc": "pass",
    }
    for percentile in (0.90, 0.95):
        selected = sweep[sweep["within_sample_percentile"] == percentile]
        assert selected["Symbol"].tolist() == ["CTA-A", "CTA-B"]
        assert selected["cumulative_fraction"].iloc[-1] == 1.0

    fractions = coverage.within_sample_percentile_addressable_fraction_by_cohort(
        "NOT_A_REAL_CODE", percentile=0.95, coverage=sweep
    )
    assert fractions.to_dict() == {"NOT_A_REAL_CODE": 1.0}


def test_within_sample_percentile_coverage_records_missing_and_zero(monkeypatch):
    panel = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["CTA_A"],
            "Symbol": ["CTA-A"],
        }
    )

    def fake_ranks(code, **kwargs):
        if code == "MISSING":
            raise FileNotFoundError("not staged")
        return panel, ["patient_1", "patient_2"], np.array([[0.1, 0.2]])

    monkeypatch.setattr(coverage, "_within_sample_percentile_panel_ranks", fake_ranks)
    result = coverage.within_sample_percentile_coverage_sweep(
        ["ZERO", "MISSING"], percentiles=(0.95,), proteoform=False, on_missing="record"
    )

    assert result[["cancer_code", "rank", "cumulative_fraction"]].to_dict("records") == [
        {"cancer_code": "ZERO", "rank": 0, "cumulative_fraction": 0.0}
    ]
    assert result.attrs["audited_cohorts"] == ["ZERO"]
    assert result.attrs["missing_cohorts"] == {"MISSING": "not staged"}


def test_within_sample_percentile_coverage_validates_inputs():
    with pytest.raises(ValueError, match="percentiles"):
        coverage.within_sample_percentile_coverage_sweep([], percentiles=(0.5,))
    with pytest.raises(ValueError, match="on_missing"):
        coverage.within_sample_percentile_coverage_sweep([], on_missing="skip")
    with pytest.raises(ValueError, match="missing required columns"):
        coverage.within_sample_percentile_addressable_fraction_by_cohort(
            percentile=0.95, coverage=pd.DataFrame({"cancer_code": ["LUAD"]})
        )
    with pytest.raises(ValueError, match="does not contain percentile"):
        coverage.within_sample_percentile_addressable_fraction_by_cohort(
            percentile=0.90, coverage=_percentile_coverage_fixture(0.95)
        )


def _percentile_coverage_fixture(percentile):
    row = coverage._zero_coverage_row("LUAD", percentile, 1)
    return pd.DataFrame([row], columns=coverage._PERCENTILE_COVERAGE_COLUMNS)


# ---- required real-data parity ----

from oncoref import source_matrices as _sm  # noqa: E402


def test_real_cohort_coverage_is_consistent():
    # The 48 MB matrix is a required test input. ``ensure`` uses the versioned
    # local/CI cache and downloads it from the source-matrix release when absent.
    assert _sm.ensure("LUAD").is_file()
    af = coverage.addressable_fraction("LUAD", threshold_tpm=10)
    pf = coverage.cta_patient_fractions("LUAD", threshold_tpm=10)
    gc = coverage.greedy_coverage("LUAD", threshold_tpm=10)
    # union >= best single CTA, <= 1, and the greedy curve converges to it.
    assert pf["fraction_expressing"].max() <= af <= 1.0
    assert abs(gc["cumulative_fraction"].iloc[-1] - af) < 1e-9
    assert (np.diff(gc["cumulative_fraction"].to_numpy()) > 0).all()
