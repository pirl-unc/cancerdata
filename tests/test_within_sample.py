# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import pandas as pd
import pytest

from oncoref import data_bundle, expression, expression_builders, source_matrices


def _matrix(genes, samples, values):
    """genes × samples DataFrame with id columns."""
    df = pd.DataFrame(values, columns=samples)
    df.insert(0, "Symbol", [f"G{i}" for i in range(len(genes))])
    df.insert(0, "Ensembl_Gene_ID", genes)
    return df


def test_within_sample_top_fractions_core():
    # 4 genes, 2 samples. In both samples gene 3 (value 100) is the top gene.
    genes = ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
    vals = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [100.0, 100.0]])
    df = _matrix(genes, ["s1", "s2"], vals)
    out = expression_builders.within_sample_top_fractions(df, thresholds=[0.95])
    # gene 4 ranks top (pct=1.0 >= 0.95) in both samples -> 1.0
    assert out.loc[out["Ensembl_Gene_ID"] == "ENSG4", "frac_samples_top5pct"].iloc[0] == 1.0
    # gene 1 is lowest -> 0.0
    assert out.loc[out["Ensembl_Gene_ID"] == "ENSG1", "frac_samples_top5pct"].iloc[0] == 0.0
    assert out["n_samples"].iloc[0] == 2


def test_within_sample_top_fractions_partial_prevalence():
    # gene is top in 1 of 2 samples -> 0.5
    genes = ["A", "B"]
    # sample s1: B is top; sample s2: A is top
    vals = np.array([[1.0, 100.0], [100.0, 1.0]])
    df = _matrix(genes, ["s1", "s2"], vals)
    out = expression_builders.within_sample_top_fractions(df, thresholds=[0.95])
    assert out["frac_samples_top5pct"].tolist() == [0.5, 0.5]


def test_within_sample_top_fractions_requires_samples():
    df = pd.DataFrame({"Ensembl_Gene_ID": ["A"], "Symbol": ["G0"]})
    with pytest.raises(ValueError, match="no per-sample columns"):
        expression_builders.within_sample_top_fractions(df)


@pytest.fixture
def within_sample_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    # Isolate the per-sample source-matrix cache too, so a cohort with no shard is
    # genuinely unresolvable (otherwise an ambient staged matrix, e.g. BRCA, lets the
    # reader recompute on the fly and the missing-cohort test stops raising).
    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path / "source-matrices"))
    shard_dir = tmp_path / "cancer-reference-expression-within-sample-top5"
    shard_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1", "ENSG2"],
            "Symbol": ["G1", "G2"],
            "frac_samples_top1pct": [0.1, 0.9],
            "frac_samples_top5pct": [0.3, 0.95],
            "frac_samples_top10pct": [0.5, 1.0],
            "n_samples": [40, 40],
        }
    ).to_parquet(shard_dir / "PRAD.parquet", index=False)
    return tmp_path


def test_available_within_sample_cohorts(within_sample_cache):
    assert expression.available_within_sample_cohorts() == ["PRAD"]


def test_locally_available_within_sample_cohorts_unions_all_local_roots(
    within_sample_cache, monkeypatch, tmp_path
):
    package_root = tmp_path / "package"
    package_shards = package_root / "cancer-reference-expression-within-sample-top5"
    package_shards.mkdir(parents=True)
    (package_shards / "LUAD.parquet").write_text("present")
    package_percentiles = package_root / "cancer-reference-expression-percentiles"
    package_percentiles.mkdir(parents=True)
    (package_percentiles / "LUAD.parquet").write_text("present")
    cached_percentiles = within_sample_cache / "cancer-reference-expression-percentiles"
    cached_percentiles.mkdir(parents=True)
    (cached_percentiles / "PRAD.parquet").write_text("present")
    monkeypatch.setattr(data_bundle, "_PACKAGE_DATA_DIR", package_root)
    monkeypatch.setattr(source_matrices, "available_cohorts", lambda: ["BRCA", "SKCM"])
    monkeypatch.setattr(source_matrices, "is_cached", lambda code: code == "BRCA")
    monkeypatch.setattr(
        data_bundle,
        "ensure_local",
        lambda: pytest.fail("local availability must not download the bundle"),
    )

    assert expression.locally_available_within_sample_cohorts() == ["BRCA", "LUAD", "PRAD"]
    assert expression.locally_available_within_sample_cohorts(include_recomputable=False) == [
        "LUAD",
        "PRAD",
    ]
    assert expression.locally_available_percentile_cohorts() == ["BRCA", "LUAD", "PRAD"]
    assert expression.locally_available_percentile_cohorts(include_recomputable=False) == [
        "LUAD",
        "PRAD",
    ]


def test_within_sample_top_fraction_default_threshold(within_sample_cache):
    df = expression.within_sample_top_fraction("PRAD")  # default 0.95 -> top5pct
    assert "frac_samples_top5pct" in df.columns
    assert df.loc[df["Symbol"] == "G2", "frac_samples_top5pct"].iloc[0] == 0.95


def test_within_sample_top_fraction_alt_threshold(within_sample_cache):
    df = expression.within_sample_top_fraction("prostate", threshold=0.90)
    assert "frac_samples_top10pct" in df.columns


def test_within_sample_bad_threshold(within_sample_cache):
    with pytest.raises(ValueError, match="threshold must be one of"):
        expression.within_sample_top_fraction("PRAD", threshold=0.5)


def test_within_sample_missing_cohort_no_fetch(within_sample_cache):
    # BRCA has no shard; must raise cleanly WITHOUT triggering a bundle download.
    with pytest.raises(ValueError, match="no within-sample"):
        expression.within_sample_top_fraction("BRCA")
