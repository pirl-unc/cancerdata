# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oncoref import data_bundle, expression, normalization

_BREAKPOINTS = [0, 1, 5, 10, 50, 90, 95, 99, 100]


def _write_percentile_shard(cache_dir, code, tpm_values):
    """Write a synthetic percentile parquet (stored as log1p, like the real one)."""
    shard_dir = cache_dir / "cancer-reference-expression-percentiles"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cols = {"Ensembl_Gene_ID": ["ENSG00000000003", "ENSG00000000005"], "Symbol": ["TSPAN6", "TNMD"]}
    for bp in _BREAKPOINTS:
        # store log1p of a per-breakpoint tpm so as_tpm restores tpm_values[bp]
        cols[f"p{bp}"] = np.log1p([tpm_values[bp], tpm_values[bp] * 2]).astype("float16")
    pd.DataFrame(cols).to_parquet(shard_dir / f"{code}.parquet", index=False)


def _write_artifact_build_metadata(cache_dir, code, *, sample_qc="pass"):
    pd.DataFrame(
        {
            "cancer_code": [code],
            "source_cohort": [code],
            "source_version": ["test"],
            "source_matrix_path": [str(cache_dir / "source.parquet")],
            "sample_qc": [sample_qc],
            "sample_qc_effective": [sample_qc],
            "sample_qc_policy_version": ["test_policy"],
            "n_source_samples": [2],
            "n_cohort_samples": [2],
            "n_qc_pass": [2 if sample_qc == "pass" else 0],
            "n_qc_warn": [0],
            "n_qc_fail": [0],
        }
    ).to_csv(cache_dir / "expression-artifact-build-metadata.csv", index=False)


def _mock_reference_summary_shards(monkeypatch, summary):
    """Serve synthetic summary rows through the bounded shard-reader boundary."""

    def load(code, sources, *, genes=None, shard_cache=None):
        source_ids = set(sources["source_cohort"].dropna().astype(str))
        mask = summary["cancer_code"].astype(str).eq(str(code))
        mask &= summary["source_cohort"].astype(str).isin(source_ids)
        selected = summary.loc[mask]
        if genes is not None and not selected.empty:
            selected = selected.loc[expression._gene_filter_mask(selected, genes)]
        return selected.copy()

    monkeypatch.setattr(expression, "_reference_summary_rows_for_sources", load)


@pytest.fixture
def percentile_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    tpm = {bp: float(bp) for bp in _BREAKPOINTS}  # p95 -> 95 tpm, etc.
    _write_percentile_shard(tmp_path, "PRAD", tpm)
    return tmp_path


def test_available_percentile_cohorts(percentile_cache):
    sidecar = percentile_cache / "cancer-reference-expression-percentiles" / "._ACC.parquet"
    sidecar.write_bytes(b"appledouble sidecar")
    assert expression.available_percentile_cohorts() == ["PRAD"]


def test_expression_artifact_gene_universe_deltas_expose_known_remaps():
    df = expression.expression_artifact_gene_universe_deltas()

    paxx = df[
        (df["legacy_ensembl_gene_id"] == "ENSG00000148362")
        & (df["oncoref_ensembl_gene_id"] == "ENSG00000310560")
    ]
    assert not paxx.empty
    assert set(paxx["symbol"]) == {"PAXX"}
    assert df.attrs["comparison"] == "pirlygenes_5.23.2_vs_oncoref_5.23.3"
    assert df.attrs["issues"] == ["#191", "#193", "#278"]
    assert {
        "gene_biotype",
        "artifact_row_class",
        "is_filterable_extra",
        "is_technical_extra",
        "is_missing_biological",
        "recommended_consumer_action",
    } <= set(df.columns)


def test_expression_artifact_gene_universe_deltas_filter_by_product_and_code():
    cll = expression.expression_artifact_gene_universe_deltas(
        product="cohort_gene_percentiles",
        cancer_type="CLL",
        delta_kind="pirlygenes_only",
    )

    assert len(cll) == 17
    assert set(cll["status"]) == {"remapped_to_oncoref"}
    assert "ENSG00000225489" in set(cll["legacy_ensembl_gene_id"])

    reps = expression.expression_artifact_gene_universe_deltas(
        product="representative_cohort_samples",
        cancer_type="PRAD",
        delta_kind="pirlygenes_only",
        status="sequence_identical_remapped_to_oncoref",
    )
    assert len(reps) == 10
    assert {
        ("ENSG00000205989", "ENSG00000206034"),
        ("ENSG00000272681", "ENSG00000279245"),
    } <= set(zip(reps["legacy_ensembl_gene_id"], reps["oncoref_ensembl_gene_id"]))


def test_expression_artifact_gene_universe_deltas_expose_full_representative_extras():
    prad = expression.expression_artifact_gene_universe_deltas(
        product="representative_cohort_samples",
        cancer_type="PRAD",
        delta_kind="oncoref_only",
    )

    assert len(prad) == 249
    assert "ENSG00000131548" in set(prad["oncoref_ensembl_gene_id"])
    assert {
        "technical_or_noncoding_extra",
        "non_signal_oncoref_extra",
        "biological_oncoref_extra",
        "y_linked_extra",
        "intentional_canonicalization",
        "unresolved_oncoref_extra",
    } <= set(prad["status"])
    technical = prad[prad["is_technical_extra"]]
    assert len(technical) == 142
    assert set(technical["artifact_row_class"]) == {"technical_extra"}
    assert set(technical["recommended_consumer_action"]) == {"filter_from_signal_views"}

    non_signal = prad[prad["artifact_row_class"].eq("non_signal_extra")]
    assert len(non_signal) == 56
    assert set(non_signal["recommended_consumer_action"]) == {"filter_from_signal_views"}
    assert non_signal["is_filterable_extra"].all()

    biological = prad[prad["artifact_row_class"].eq("biological_extra")]
    assert len(biological) == 45
    assert set(biological["recommended_consumer_action"]) == {"keep_oncoref_biological_row"}
    assert not biological["is_filterable_extra"].any()


def test_expression_artifact_gene_universe_deltas_flag_technical_and_missing_rows():
    df = expression.expression_artifact_gene_universe_deltas()

    technical = df[df["is_technical_extra"]]
    assert len(technical) == 563
    assert set(technical["status"]) == {
        "technical_or_noncoding_extra",
        "y_linked_extra",
        "immune_receptor_segment_extra",
    }
    assert set(technical["artifact_row_class"]) == {"technical_extra"}
    assert set(technical["recommended_consumer_action"]) == {"filter_from_signal_views"}

    non_signal = df[df["artifact_row_class"].eq("non_signal_extra")]
    assert len(non_signal) == 237
    assert non_signal["is_filterable_extra"].all()
    assert not non_signal["is_technical_extra"].any()
    assert set(non_signal["status"]) == {"non_signal_oncoref_extra"}
    assert set(non_signal["recommended_consumer_action"]) == {"filter_from_signal_views"}

    biological = df[df["artifact_row_class"].eq("biological_extra")]
    assert len(biological) == 196
    assert not biological["is_filterable_extra"].any()
    assert set(biological["status"]) == {"biological_oncoref_extra"}
    assert set(biological["gene_biotype"]) == {"lncRNA", "protein_coding"}
    assert set(biological["recommended_consumer_action"]) == {"keep_oncoref_biological_row"}

    unresolved = df[df["artifact_row_class"].eq("unresolved_extra")]
    assert len(unresolved) == 29
    assert set(unresolved["status"]) == {"unresolved_oncoref_extra"}
    assert set(unresolved["gene_biotype"].dropna()) == set()
    assert set(unresolved["recommended_consumer_action"]) == {"audit_before_filtering"}

    missing = df[df["is_missing_biological"]]
    assert missing.empty

    sequence = df[df["status"].eq("sequence_identical_remapped_to_oncoref")]
    assert len(sequence) == 32
    assert set(sequence["artifact_row_class"]) == {"canonicalized"}
    assert set(sequence["recommended_consumer_action"]) == {"accept_canonical_mapping"}


def test_expression_artifact_technical_extra_gene_ids_filters_request_scope():
    ids = expression.expression_artifact_technical_extra_gene_ids(
        product="representative_cohort_samples",
        cancer_type="PRAD",
    )

    assert len(ids) == 142
    assert ids == sorted(ids)
    assert "ENSG00000199334" in ids
    assert "ENSG00000310560" not in ids


def test_representatives_filter_and_flag_artifact_technical_extras(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000199334", "ENSG00000000003"],
            "Symbol": ["RNA5S11", "TSPAN6"],
            "PRAD__rep1": [1.0, 2.0],
        }
    ).to_parquet(d / "PRAD.parquet", index=False)

    flagged = expression.representative_cohort_samples("PRAD", include_gene_universe_flags=True)
    keyed = flagged.set_index("Ensembl_Gene_ID")
    assert keyed.loc["ENSG00000199334", "artifact_row_class"] == "technical_extra"
    assert bool(keyed.loc["ENSG00000199334", "is_technical_extra"]) is True
    assert keyed.loc["ENSG00000199334", "recommended_consumer_action"] == (
        "filter_from_signal_views"
    )
    assert keyed.loc["ENSG00000000003", "artifact_row_class"] == "artifact"
    assert flagged.attrs["gene_universe"] == "artifact"
    assert flagged.attrs["include_gene_universe_flags"] is True

    filtered = expression.representative_cohort_samples("PRAD", gene_universe="tumor_signal")
    assert list(filtered["Ensembl_Gene_ID"]) == ["ENSG00000000003"]
    assert filtered.attrs["gene_universe"] == "tumor_signal"

    long = expression.representative_cohort_samples(
        "PRAD",
        format="long",
        gene_universe="tumor_signal",
        include_gene_universe_flags=True,
    )
    assert list(long["Ensembl_Gene_ID"]) == ["ENSG00000000003"]
    assert "artifact_row_class" in long.columns


def test_representatives_pirlygenes_universe_keeps_remaps_and_drops_audited_extras(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": [
                "ENSG00000199334",  # technical extra
                "ENSG00000206034",  # sequence-identical pirlygenes remap target
                "ENSG00000178287",  # biological oncoref-only extra
                "ENSG00000261190",  # unresolved oncoref-only extra
                "ENSG00000000003",  # shared artifact row
            ],
            "Symbol": ["RNA5S11", "DEFB109B", "SPAG11A", "LINC02911", "TSPAN6"],
            "PRAD__rep1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).to_parquet(d / "PRAD.parquet", index=False)

    tumor_signal = expression.representative_cohort_samples(
        "PRAD", gene_universe="tumor_signal", gene_id_style="pirlygenes"
    )
    pirlygenes = expression.representative_cohort_samples(
        "PRAD", gene_universe="pirlygenes", gene_id_style="pirlygenes"
    )

    assert set(tumor_signal["Ensembl_Gene_ID"]) == {
        "ENSG00000205989",
        "ENSG00000178287",
        "ENSG00000261190",
        "ENSG00000000003",
    }
    assert set(pirlygenes["Ensembl_Gene_ID"]) == {
        "ENSG00000205989",
        "ENSG00000000003",
    }
    remapped = pirlygenes.set_index("Ensembl_Gene_ID").loc["ENSG00000205989"]
    assert remapped["Symbol"] == "DEFB109C"
    assert remapped["PRAD_rep01"] == pytest.approx(2.0)


def test_representatives_pirlygenes_universe_matches_known_parity_counts():
    expected = {
        "PRAD": 34337,
        "CLL": 51796,
        "COAD_MSI": 34337,
        "READ_MSI": 34337,
    }

    for code, count in expected.items():
        out = expression.representative_cohort_samples(
            code,
            k=1,
            format="long",
            gene_universe="pirlygenes",
            gene_id_style="pirlygenes",
        )
        assert len(out) == count
        assert out["Ensembl_Gene_ID"].nunique() == count
        assert out.attrs["gene_universe"] == "pirlygenes"
        if code == "CLL":
            assert {"ENSG00000226079", "ENSG00000232395"} <= set(out["Ensembl_Gene_ID"])


def test_stad_ucec_subtype_expression_artifacts_ship():
    expected = {
        "STAD_CIN",
        "STAD_MSI",
        "STAD_GS",
        "STAD_EBV",
        "UCEC_CNH",
        "UCEC_MSI",
        "UCEC_CNL",
        "UCEC_POLE",
    }

    assert expected <= set(expression.available_representative_cohorts())
    assert expected <= set(expression.available_percentile_cohorts())

    metadata = expression.expression_artifact_build_metadata(expected)
    assert set(metadata["cancer_code"]) == expected
    assert set(metadata["source_cohort"]) == {
        "TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE",
        "TREEHOUSE_POLYA_25_01_TCGA_UCEC_SUBTYPE",
    }
    assert metadata.set_index("cancer_code").loc["UCEC_CNH", "n_source_samples"] == 85
    assert metadata.set_index("cancer_code").loc["UCEC_CNH", "n_cohort_samples"] == 83


def test_mbl_molecular_subgroup_expression_artifacts_ship_with_provenance():
    expected_source_samples = {
        "MBL_WNT": 17,
        "MBL_SHH": 25,
        "MBL_G3": 44,
        "MBL_G4": 39,
    }
    expected_reference_samples = {
        "MBL_WNT": 4,
        "MBL_SHH": 7,
        "MBL_G3": 29,
        "MBL_G4": 18,
    }
    codes = set(expected_source_samples)

    assert codes <= set(expression.available_representative_cohorts())
    assert codes <= set(expression.available_percentile_cohorts())
    assert codes <= set(expression.available_within_sample_cohorts())

    metadata = expression.expression_artifact_build_metadata(
        codes, auto_fetch=False, on_missing="raise"
    ).set_index("cancer_code")
    availability = expression.cancer_reference_expression_availability(
        sorted(codes),
        reference_source="artifact",
        sample_qc="artifact",
    ).set_index("cancer_code")

    assert metadata["n_source_samples"].to_dict() == expected_source_samples
    assert metadata["n_cohort_samples"].to_dict() == expected_reference_samples
    assert set(metadata["sample_qc_effective"]) == {"pass"}
    assert set(metadata["source_cohort"]) == {"TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS"}
    assert availability["available"].all()
    assert availability["n_reference_samples"].to_dict() == expected_reference_samples
    assert availability["source_project"].tolist() == ["Treehouse"] * 4
    assert availability["source_type"].tolist() == ["treehouse-derived"] * 4
    assert availability["source_scale_class"].tolist() == ["linear_rnaseq_tpm"] * 4
    assert availability["linear_tpm_comparable"].all()
    assert availability["processing_pipeline"].notna().all()


def test_sparse_source_qc_release_artifacts_have_count_and_diagnostic_parity():
    expected = {
        "BL": {"source": 184, "pass": 175, "fail": 9},
        "SARC_PEC": {"source": 69, "pass": 60, "fail": 9},
    }
    codes = list(expected)

    metadata = expression.expression_artifact_build_metadata(
        codes, auto_fetch=False, on_missing="raise"
    ).set_index("cancer_code")
    summary_availability = expression.cancer_reference_expression_availability(
        codes,
        reference_source="summary_rows_all",
        sample_qc="all",
    ).set_index("cancer_code")
    artifact_availability = expression.cancer_reference_expression_availability(
        codes,
        reference_source="artifact",
        sample_qc="artifact",
    ).set_index("cancer_code")
    artifact_reference = expression.cancer_reference_expression(
        codes,
        genes="ENSG00000141510",
        reference_source="artifact",
        sample_qc="artifact",
    )

    provenance_path = data_bundle.find(
        "cancer-reference-expression-representatives/_provenance.csv"
    )
    assert provenance_path is not None
    provenance = pd.read_csv(provenance_path)
    provenance_codes = provenance["representative_id"].astype(str).str.split("__").str[0]

    for code, counts in expected.items():
        assert metadata.loc[code, "n_source_samples"] == counts["source"]
        assert metadata.loc[code, "n_cohort_samples"] == counts["pass"]
        assert metadata.loc[code, "n_qc_pass"] == counts["pass"]
        assert metadata.loc[code, "n_qc_fail"] == counts["fail"]
        assert metadata.loc[code, "sample_qc_effective"] == "pass"
        assert summary_availability.loc[code, "n_reference_samples"] == counts["pass"]
        assert artifact_availability.loc[code, "n_reference_samples"] == counts["pass"]
        reference_rows = artifact_reference.loc[artifact_reference["cancer_code"] == code]
        assert not reference_rows.empty
        assert set(reference_rows["n_reference_samples"]) == {counts["pass"]}
        assert set(reference_rows["n_samples"]) == {counts["pass"]}
        assert set(provenance.loc[provenance_codes.eq(code), "n_cohort_samples"]) == {
            counts["pass"]
        }
        assert set(expression.within_sample_top_fraction(code)["n_samples"]) == {counts["pass"]}
        assert not expression.cohort_gene_percentiles(code).empty

    diagnostics = {
        "JNGR150": "low_detected_genes;low_housekeeping_detection;high_zero_fraction",
        "JNGR175": "high_zero_fraction",
        "JNGR178": "high_top_gene_fraction",
        "JNGR179": "low_detected_genes;high_zero_fraction",
        "BLGSP-71-23-00437-01A": "high_zero_fraction",
    }
    qc = expression.source_matrix_sample_qc_manifest(
        codes, auto_fetch=False, on_missing="raise"
    ).set_index("sample_id")
    for sample_id, reasons in diagnostics.items():
        assert qc.loc[sample_id, "sample_qc_status"] == "fail"
        assert qc.loc[sample_id, "sample_qc_reasons"] == reasons
        assert qc.loc[sample_id, "parse_missing_fraction"] == 0.0
        assert qc.loc[sample_id, "zero_fraction_raw"] > 0.0
    assert qc.loc["JNGR150", "n_detected_genes"] == 1223


def test_expression_artifact_gene_universe_delta_summary():
    summary = expression.expression_artifact_gene_universe_delta_summary()

    hit = summary[
        (summary["product"] == "cohort_gene_percentiles")
        & (summary["cancer_code"] == "CLL")
        & (summary["delta_kind"] == "pirlygenes_only")
        & (summary["status"] == "remapped_to_oncoref")
    ]
    assert hit["n"].iloc[0] == 17
    assert hit["artifact_row_class"].iloc[0] == "canonicalized"
    assert not hit["is_missing_biological"].iloc[0]

    representative = summary[
        (summary["product"] == "representative_cohort_samples")
        & (summary["cancer_code"] == "PRAD")
        & (summary["delta_kind"] == "oncoref_only")
    ]
    assert representative["n"].sum() == 249


def test_expression_artifact_gene_universe_delta_report_scopes_requests():
    report = expression.expression_artifact_gene_universe_delta_report(
        "representative_cohort_samples", "PRAD"
    )

    assert report.attrs["comparison"] == "pirlygenes_5.23.2_vs_oncoref_5.23.3"
    assert report.attrs["issues"] == ["#191", "#193", "#278"]
    assert report.attrs["requested_cancer_codes"] == ["PRAD"]
    assert int(report["n"].sum()) == 264
    assert {
        "technical_or_noncoding_extra",
        "sequence_identical_remapped_to_oncoref",
        "remapped_to_oncoref",
    } <= set(report["status"])
    assert (
        report.loc[
            report["status"].eq("technical_or_noncoding_extra"), "recommended_consumer_action"
        ]
        .eq("filter_from_signal_views")
        .all()
    )

    ref_report = expression.expression_artifact_gene_universe_delta_report(
        "cancer_reference_expression", "CLL"
    )
    assert set(ref_report["product"]) == {"cohort_gene_percentiles"}
    assert int(ref_report["n"].sum()) == 20

    empty = expression.expression_artifact_gene_universe_delta_report("cohort_gene_percentiles", [])
    assert empty.empty
    assert list(empty.columns) == [
        "accessor",
        "product",
        "cancer_code",
        "delta_kind",
        "status",
        "artifact_row_class",
        "is_filterable_extra",
        "is_technical_extra",
        "is_missing_biological",
        "recommended_consumer_action",
        "n",
        "legacy_ensembl_gene_ids",
        "oncoref_ensembl_gene_ids",
        "gene_biotypes",
        "symbols",
        "issues",
    ]


def test_cohort_gene_percentiles_as_tpm(percentile_cache):
    df = expression.cohort_gene_percentiles("PRAD", as_tpm=True)
    assert list(df["Symbol"]) == ["TSPAN6", "TNMD"]
    # gene 0, p95 breakpoint should restore to ~95 tpm (stored as log1p(95)).
    assert df.loc[0, "p95"] == pytest.approx(95.0, rel=1e-2)
    assert df.attrs["gene_universe_delta_n"] == 10
    assert df.attrs["gene_universe_delta_issues"] == ["#191", "#193", "#278"]
    assert df.attrs["sample_qc"] == "pass"
    assert df.attrs["artifact_sample_qc_verified"] is False


def test_cohort_gene_percentiles_log_space(percentile_cache):
    df = expression.cohort_gene_percentiles("PRAD", as_tpm=False)
    # stored log1p value, not expm1-restored
    assert df.loc[0, "p95"] == pytest.approx(np.log1p(95.0), rel=1e-2)


def test_cohort_gene_percentiles_resolves_alias(percentile_cache):
    # "prostate" resolves to PRAD via the registry.
    df = expression.cohort_gene_percentiles("prostate")
    assert len(df) == 2


def test_cohort_gene_percentiles_can_return_pirlygenes_legacy_gene_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    shard_dir = tmp_path / "cancer-reference-expression-percentiles"
    shard_dir.mkdir(parents=True)
    row = {
        "Ensembl_Gene_ID": "ENSG00000310560",
        "Symbol": "PAXX",
    }
    for bp in _BREAKPOINTS:
        row[f"p{bp}"] = np.log1p(float(bp))
    pd.DataFrame([row]).to_parquet(shard_dir / "PRAD.parquet", index=False)

    canonical = expression.cohort_gene_percentiles("PRAD")
    legacy = expression.cohort_gene_percentiles("PRAD", gene_id_style="pirlygenes")

    assert canonical.loc[0, "Ensembl_Gene_ID"] == "ENSG00000310560"
    assert canonical.attrs["gene_id_style"] == "oncoref"
    assert legacy.loc[0, "Ensembl_Gene_ID"] == "ENSG00000148362"
    assert legacy.loc[0, "Symbol"] == "PAXX"
    assert legacy.loc[0, "p95"] == pytest.approx(canonical.loc[0, "p95"])
    assert legacy.attrs["gene_id_style"] == "pirlygenes"


def test_cohort_gene_percentiles_rejects_pirlygenes_ids_for_proteoform_artifacts():
    with pytest.raises(ValueError, match="gene-level artifacts"):
        expression.cohort_gene_percentiles("PRAD", proteoform=True, gene_id_style="pirlygenes")


def _no_cached_matrix(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(expression, "per_sample_expression", _raise)


def _mock_missing_artifact_build_metadata(monkeypatch):
    missing = pd.DataFrame()
    missing.attrs["missing_reason"] = "artifact_build_metadata_not_installed"
    monkeypatch.setattr(
        expression,
        "expression_artifact_build_metadata",
        lambda *args, **kwargs: missing.copy(),
    )


def test_cohort_gene_percentiles_missing_raises(percentile_cache, monkeypatch):
    # No shard AND no cached matrix (on-the-fly can't run) -> clear error.
    _no_cached_matrix(monkeypatch)
    with pytest.raises(ValueError, match="no percentile vector"):
        expression.cohort_gene_percentiles("BRCA")


def test_cohort_gene_percentiles_missing_empty_schema(percentile_cache, monkeypatch):
    _no_cached_matrix(monkeypatch)
    df = expression.cohort_gene_percentiles("BRCA", on_missing="empty", include_provenance=True)

    assert df.empty
    assert list(df.columns[:2]) == ["Ensembl_Gene_ID", "Symbol"]
    assert "p0" in df.columns and "p100" in df.columns
    assert "cancer_code" in df.columns
    assert df.attrs["schema_version"] == expression.PERCENTILE_ARTIFACT_SCHEMA_VERSION
    assert df.attrs["cancer_code"] == "BRCA"
    assert "missing_reason" in df.attrs


def test_cohort_gene_percentiles_provenance_columns_and_attrs(percentile_cache):
    _write_artifact_build_metadata(percentile_cache, "PRAD", sample_qc="pass")
    df = expression.cohort_gene_percentiles("PRAD", include_provenance=True)

    assert set(df.columns) >= {
        "cancer_code",
        "normalization",
        "expression_unit",
        "percentile_basis",
        "artifact_schema_version",
        "data_version",
        "source_matrix_version",
        "sample_qc",
        "sample_qc_policy_version",
    }
    assert set(df["cancer_code"]) == {"PRAD"}
    assert set(df["normalization"]) == {"tpm_clean"}
    assert set(df["sample_qc"]) == {"pass"}
    assert set(df["sample_qc_policy_version"]) == {"test_policy"}
    assert df.attrs["schema_version"] == expression.PERCENTILE_ARTIFACT_SCHEMA_VERSION
    assert df.attrs["cancer_code"] == "PRAD"
    assert df.attrs["artifact_sample_qc"] == "pass"
    assert df.attrs["artifact_sample_qc_verified"] is True


def test_cohort_gene_percentiles_rejects_mismatched_artifact_sample_qc(percentile_cache):
    _write_artifact_build_metadata(percentile_cache, "PRAD", sample_qc="all")

    with pytest.raises(ValueError, match="sample_qc mismatch"):
        expression.cohort_gene_percentiles("PRAD")

    audit = expression.cohort_gene_percentiles("PRAD", sample_qc="artifact")
    assert audit.attrs["sample_qc"] == "artifact"
    assert audit.attrs["artifact_sample_qc"] == "all"


@pytest.fixture
def proteoform_percentile_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    shard_dir = tmp_path / "cancer-reference-expression-percentiles-proteoform-cta"
    shard_dir.mkdir(parents=True)
    # A collapsed shard carries the proteoform identity columns alongside the breakpoints.
    cols = {
        "proteoform_key": ["NY-ESO-1", "ENSG00000185686"],
        "Ensembl_Gene_ID": ["ENSG00000184033", "ENSG00000185686"],
        "Symbol": ["NY-ESO-1", "PRAME"],
        "proteoform_members": ["CTAG1A/CTAG1B", "PRAME"],
    }
    for bp in _BREAKPOINTS:
        cols[f"p{bp}"] = np.log1p([float(bp), float(bp) * 2]).astype("float16")
    pd.DataFrame(cols).to_parquet(shard_dir / "PRAD.parquet", index=False)
    return tmp_path


def test_available_percentile_cohorts_proteoform(proteoform_percentile_cache):
    # The proteoform variant reads the local shard (auto_fetch=False, no bundle fetch).
    assert expression.available_percentile_cohorts(proteoform=True) == ["PRAD"]


def test_cohort_gene_percentiles_proteoform(proteoform_percentile_cache):
    df = expression.cohort_gene_percentiles("PRAD", as_tpm=True, proteoform=True)
    # the proteoform key space: NY-ESO-1 (collapsed) + PRAME (singleton -> ENSG key)
    assert list(df["proteoform_key"]) == ["NY-ESO-1", "ENSG00000185686"]
    # breakpoint columns are restored to TPM; the id columns are excluded from them
    assert df.loc[0, "p95"] == pytest.approx(95.0, rel=1e-2)
    assert set(df.columns) >= {"proteoform_key", "Ensembl_Gene_ID", "Symbol", "proteoform_members"}


def test_cohort_gene_percentiles_proteoform_missing_raises(
    proteoform_percentile_cache, monkeypatch
):
    _no_cached_matrix(monkeypatch)
    with pytest.raises(ValueError, match="no proteoform-summed percentile vector"):
        expression.cohort_gene_percentiles("BRCA", proteoform=True)


def test_cohort_gene_percentiles_proteoform_computed_on_the_fly(
    proteoform_percentile_cache, monkeypatch
):
    # LUAD has no proteoform shard -> the percentile vector is recomputed on the fly
    # from the (stubbed) per-sample matrix: members collapse before the percentiles.
    import oncoref.proteoforms as pmod

    fake = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG_A1", "ENSG_A2", "ENSG_B"],
            "Symbol": ["A1", "A2", "B"],
            "s1": [3.0, 5.0, 1.0],
            "s2": [9.0, 1.0, 4.0],
        }
    )
    monkeypatch.setattr(expression, "per_sample_expression", lambda code, **k: fake.copy())
    monkeypatch.setattr(
        pmod, "proteoform_group_map", lambda *, scope="cta": {"A1/A2": ["ENSG_A1", "ENSG_A2"]}
    )
    out = expression.cohort_gene_percentiles("LUAD", proteoform=True)
    assert "proteoform_key" in out.columns
    assert "A1/2" in set(out["Symbol"]) and "A1" not in set(out["Symbol"])
    # A1/A2 summed per sample: s1=8, s2=10 -> p100 (max) restores to 10.0 TPM
    assert out.set_index("Symbol").loc["A1/2", "p100"] == pytest.approx(10.0, rel=1e-2)


def test_cohort_gene_percentiles_threads_scope(proteoform_percentile_cache, monkeypatch):
    # scope= reaches the on-the-fly collapse: the cta universe leaves A1/A2 separate,
    # the genome universe groups them. Previously scope was silently fixed.
    import oncoref.proteoforms as pmod

    fake = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG_A1", "ENSG_A2", "ENSG_B"],
            "Symbol": ["A1", "A2", "B"],
            "s1": [3.0, 5.0, 1.0],
            "s2": [9.0, 1.0, 4.0],
        }
    )
    monkeypatch.setattr(expression, "per_sample_expression", lambda code, **k: fake.copy())
    maps = {"cta": {}, "genome": {"A1/A2": ["ENSG_A1", "ENSG_A2"]}}
    monkeypatch.setattr(pmod, "proteoform_group_map", lambda *, scope="cta": maps[scope])

    cta = expression.cohort_gene_percentiles("LUAD", proteoform=True, scope="cta", auto_fetch=False)
    assert {"A1", "A2"} <= set(cta["Symbol"])  # cta universe: ungrouped here
    genome = expression.cohort_gene_percentiles(
        "LUAD", proteoform=True, scope="genome", auto_fetch=False
    )
    assert "A1/2" in set(genome["Symbol"]) and "A1" not in set(genome["Symbol"])  # collapsed


def test_proteoform_shard_selection_is_scope_specific(proteoform_percentile_cache, monkeypatch):
    # The fixture ships a *cta-scope* proteoform shard. A cta request reads it; a genome
    # request must NOT (it resolves the genome dir, which is absent) — proving scope
    # selects the shard directory, not just the on-the-fly collapse universe.
    cta = expression.cohort_gene_percentiles("PRAD", proteoform=True, scope="cta", auto_fetch=False)
    assert list(cta["proteoform_key"]) == ["NY-ESO-1", "ENSG00000185686"]

    def _no_matrix(*a, **k):
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(expression, "per_sample_expression", _no_matrix)
    with pytest.raises(ValueError, match="per-sample matrix isn't cached"):
        expression.cohort_gene_percentiles(
            "PRAD", proteoform=True, scope="genome", auto_fetch=False
        )


def test_shard_dataset_registry_is_public_and_scope_aware():
    # ShardDataset + SHARD_DATASETS are public; the proteoform shard dir is scope-specific.
    import oncoref

    assert set(oncoref.SHARD_DATASETS) == {"representatives", "percentiles", "within_sample"}
    pct = oncoref.SHARD_DATASETS["percentiles"]
    assert isinstance(pct, oncoref.ShardDataset)
    assert pct.subdir(proteoform=False) == "cancer-reference-expression-percentiles"
    assert pct.subdir(proteoform=True, scope="cta").endswith("-proteoform-cta")
    assert pct.subdir(proteoform=True, scope="genome").endswith("-proteoform-genome")
    assert pct.fetches(proteoform=False) is True
    assert pct.fetches(proteoform=True, scope="cta") is True
    assert pct.fetches(proteoform=True, scope="genome") is False
    assert oncoref.SHARD_DATASETS["within_sample"].fetches(proteoform=False) is True
    assert oncoref.SHARD_DATASETS["within_sample"].fetches(proteoform=True, scope="cta") is True
    assert oncoref.SHARD_DATASETS["within_sample"].fetches(proteoform=True, scope="genome") is False
    # an artifact with no proteoform variant rejects the request
    with pytest.raises(ValueError, match="no proteoform variant"):
        oncoref.SHARD_DATASETS["representatives"].subdir(proteoform=True)
    # the column-vocabulary helpers are now package-public too
    assert callable(oncoref.id_columns) and callable(oncoref.sample_columns)


def test_proteoform_summary_wrappers_thread_scope_and_default_autofetch_true(monkeypatch):
    # Proteoform wrappers still default auto_fetch=True and thread scope; if the
    # active bundle lacks a requested scope-specific shard, they can recompute
    # from the source matrix through the same path.
    seen = {}
    monkeypatch.setattr(
        expression, "cohort_gene_percentiles", lambda ct, **k: seen.update(k) or pd.DataFrame()
    )
    expression.proteoform_cohort_percentiles("X", scope="genome")
    assert seen["proteoform"] is True and seen["scope"] == "genome" and seen["auto_fetch"] is True

    seen.clear()
    monkeypatch.setattr(
        expression, "within_sample_top_fraction", lambda ct, **k: seen.update(k) or pd.DataFrame()
    )
    expression.proteoform_within_sample_top_fraction("X", scope="genome")
    assert seen["proteoform"] is True and seen["scope"] == "genome" and seen["auto_fetch"] is True
    # The gene variant still defaults auto_fetch=False (reads a shipped shard).
    seen.clear()
    monkeypatch.setattr(
        expression, "cohort_gene_percentiles", lambda ct, **k: seen.update(k) or pd.DataFrame()
    )
    expression.gene_cohort_percentiles("X")
    assert seen["auto_fetch"] is False


def test_representatives_provenance_requires_long_format():
    # Provenance is per-representative; asking for it in the default wide form is
    # a no-op, so it must fail loudly rather than silently dropping the request.
    with pytest.raises(ValueError, match="include_provenance=True requires format='long'"):
        expression.representative_cohort_samples("PRAD", include_provenance=True)


def test_representative_provenance_includes_source_sample_and_selection_metadata(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1"],
            "Symbol": ["GENE1"],
            "PRAD__rep1": [1.0],
        }
    ).to_parquet(d / "PRAD.parquet", index=False)
    pd.DataFrame(
        {
            "representative_id": ["PRAD__rep1"],
            "source_cohort": ["PRAD"],
            "source_project": ["TCGA"],
            "source_diagnosis": ["Adenocarcinoma, NOS"],
            "source_morphology": ["8140/3"],
            "source_sample": ["TCGA-XX-0001"],
            "source_group_id": ["TCGA:TCGA-XX-0001"],
            "sample_qc": ["pass"],
            "sample_qc_requested": ["pass"],
            "source_sample_qc": ["pass"],
            "representative_role": ["standard"],
            "benchmark_eligible": [True],
            "partition_role": ["validation"],
            "partition_reason": ["stable_source_group_validation"],
            "partition_policy_version": [expression.REPRESENTATIVE_PARTITION_POLICY_VERSION],
            "partition_validation_target": [1],
            "partition_status": ["available"],
            "n_partition_train": [1],
            "n_partition_validation": [1],
            "n_partition_validation_external": [0],
            "n_partition_audit_only": [0],
            "review_source": ["https://example.test/review"],
            "review_note": ["reviewed source"],
            "n_cohort_samples": [10],
        }
    ).to_csv(d / "_provenance.csv", index=False)

    df = expression.representative_cohort_samples("PRAD", format="long", include_provenance=True)

    assert df.loc[0, "representative_id"] == "PRAD_rep01"
    assert df.loc[0, "source_sample"] == "TCGA-XX-0001"
    assert df.loc[0, "source_group_id"] == "TCGA:TCGA-XX-0001"
    assert df.loc[0, "source_diagnosis"] == "Adenocarcinoma, NOS"
    assert df.loc[0, "source_morphology"] == "8140/3"
    assert df.loc[0, "source_sample_qc"] == "pass"
    assert df.loc[0, "representative_role"] == "standard"
    assert bool(df.loc[0, "benchmark_eligible"]) is True
    assert df.loc[0, "partition_role"] == "validation"
    assert (
        df.loc[0, "partition_policy_version"] == expression.REPRESENTATIVE_PARTITION_POLICY_VERSION
    )
    assert df.loc[0, "review_source"] == "https://example.test/review"
    assert df.loc[0, "review_note"] == "reviewed source"
    assert df.loc[0, "selection_rank"] == 1
    assert df.loc[0, "selection_method"] == expression.REPRESENTATIVE_SELECTION_METHOD
    assert df.loc[0, "selection_basis"] == expression.REPRESENTATIVE_SELECTION_BASIS
    assert df.loc[0, "artifact_schema_version"] == expression.REPRESENTATIVE_ARTIFACT_SCHEMA_VERSION
    assert df.attrs["schema_version"] == expression.REPRESENTATIVE_ARTIFACT_SCHEMA_VERSION
    assert df.attrs["cancer_codes"] == ["PRAD"]
    assert df.attrs["representative_id_style"] == "pirlygenes"

    internal = expression.representative_cohort_samples(
        "PRAD",
        format="long",
        include_provenance=True,
        representative_id_style="internal",
    )
    assert internal.loc[0, "representative_id"] == "PRAD__rep1"


def test_representative_partition_manifest_is_one_row_per_vector(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    shard_dir = tmp_path / "cancer-reference-expression-representatives"
    shard_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "representative_id": ["PRAD__rep1", "PRAD__rep2", "BRCA__rep1"],
            "source_group_id": ["TCGA:p1", "TCGA:p2", "TCGA:b1"],
            "source_cohort": ["TCGA_PRAD", "TCGA_PRAD", "TCGA_BRCA"],
            "source_project": ["TCGA", "TCGA", "TCGA"],
            "source_sample": ["p1", "p2", "b1"],
            "representative_role": ["standard", "standard", "standard"],
            "benchmark_eligible": [True, True, True],
            "partition_role": ["train", "validation", "train"],
            "partition_reason": [
                "stable_source_group_training",
                "stable_source_group_validation",
                "stable_source_group_training",
            ],
            "partition_policy_version": [expression.REPRESENTATIVE_PARTITION_POLICY_VERSION] * 3,
            "partition_validation_target": [1, 1, 0],
            "partition_status": [
                "available",
                "available",
                "insufficient_independent_groups",
            ],
            "n_partition_train": [1, 1, 1],
            "n_partition_validation": [1, 1, 0],
            "n_partition_validation_external": [0, 0, 0],
            "n_partition_audit_only": [0, 0, 0],
        }
    ).to_csv(shard_dir / "_provenance.csv", index=False)

    manifest = expression.representative_partition_manifest("PRAD")

    assert manifest["representative_id"].tolist() == ["PRAD_rep01", "PRAD_rep02"]
    assert manifest["partition_role"].tolist() == ["train", "validation"]
    assert manifest["source_group_id"].is_unique
    assert (
        manifest.attrs["partition_policy_version"]
        == expression.REPRESENTATIVE_PARTITION_POLICY_VERSION
    )


def test_released_representative_partition_is_source_grouped_and_stratified():
    manifest = expression.representative_partition_manifest()
    availability = expression.representative_cohort_availability()

    assert not manifest.empty
    assert set(manifest["partition_role"]) <= {
        "train",
        "validation",
        "validation_external",
        "audit_only",
    }
    assert manifest.groupby("source_group_id")["partition_role"].nunique().max() == 1
    assert set(manifest["partition_policy_version"]) == {
        expression.REPRESENTATIVE_PARTITION_POLICY_VERSION
    }

    ordinary = availability[
        (availability["n_representatives"] == 5) & (availability["n_partition_audit_only"] == 0)
    ]
    assert not ordinary.empty
    assert set(ordinary["n_partition_train"]) == {3}
    assert set(ordinary["n_partition_validation"]) == {2}
    assert set(ordinary["partition_status"]) == {"available"}
    partition_totals = (
        availability["n_partition_train"]
        + availability["n_partition_validation"]
        + availability["n_partition_audit_only"]
    )
    assert partition_totals.equals(availability["n_representatives"])

    by_code = availability.set_index("cancer_code")
    assert by_code.loc["SARC_EHE", "partition_status"] == ("insufficient_independent_groups")
    assert by_code.loc["SARC_EHE", "n_partition_train"] == 1
    for code in ("RB", "SARC_CHOR"):
        assert by_code.loc[code, "partition_status"] == ("no_benchmark_eligible_groups")
        assert by_code.loc[code, "n_partition_audit_only"] == 5

    build_summary = expression.expression_artifact_build_summary(on_missing="raise")
    assert build_summary["representative_partition"]["grouping_key"] == ("source_group_id")
    assert build_summary["representative_partition"]["policy_version"] == (
        expression.REPRESENTATIVE_PARTITION_POLICY_VERSION
    )


def test_representative_availability_routes_proxy_and_qc_fallback_cohorts(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    rows = []
    qc_rows = []
    fixtures = {
        "MTC": {
            "sample": "GSM810624",
            "source": "GSE32662_PRINGLE_2012",
            "status": "warn",
            "reasons": "nonlinear_or_proxy_expression_scale",
            "scale": "microarray_tpm_proxy",
            "linear": False,
            "floor": False,
            "effective": "pass_or_warn",
            "fallback": "no_pass_samples_tpm_proxy_source",
            "role": "standard",
            "benchmark": True,
        },
        "RB": {
            "sample": "THR24_3114_S01",
            "source": "TREEHOUSE_RIBOD_25_01",
            "status": "fail",
            "reasons": "high_top10_gene_fraction",
            "scale": "linear_rnaseq_tpm",
            "linear": True,
            "floor": False,
            "effective": "all",
            "fallback": "no_pass_samples_high_concentration_source",
            "role": "source_qc_fallback_audit_only",
            "benchmark": False,
        },
        "PRAD": {
            "sample": "TCGA-XX-0001",
            "source": "TCGA_PRAD",
            "status": "pass",
            "reasons": "",
            "scale": "linear_rnaseq_tpm",
            "linear": True,
            "floor": True,
            "effective": "pass",
            "fallback": "",
            "role": "standard",
            "benchmark": True,
        },
    }
    for code, fixture in fixtures.items():
        pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["ENSG1"],
                "Symbol": ["GENE1"],
                f"{code}__rep1": [1.0],
            }
        ).to_parquet(d / f"{code}.parquet", index=False)
        rows.append(
            {
                "representative_id": f"{code}__rep1",
                "source_cohort": fixture["source"],
                "source_sample": fixture["sample"],
                "sample_qc_requested": "pass",
                "sample_qc_effective": fixture["effective"],
                "sample_qc_fallback_reason": fixture["fallback"],
                "sample_qc_policy_version": "test_policy",
                "representative_role": fixture["role"],
                "benchmark_eligible": fixture["benchmark"],
            }
        )
        qc_rows.append(
            {
                "cancer_code": code,
                "source_cohort": fixture["source"],
                "sample_id": fixture["sample"],
                "sample_qc_status": fixture["status"],
                "sample_qc_reasons": fixture["reasons"],
                "source_scale_class": fixture["scale"],
                "linear_tpm_comparable": fixture["linear"],
                "recommended_for_absolute_tpm_floor": fixture["floor"],
            }
        )
    pd.DataFrame(rows).to_csv(d / "_provenance.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(
        tmp_path / expression.SOURCE_MATRIX_SAMPLE_QC_MANIFEST_PATH, index=False
    )

    availability = expression.representative_cohort_availability()
    keyed = availability.set_index("cancer_code")

    assert keyed.loc["MTC", "source_scale_class"] == "microarray_tpm_proxy"
    assert not keyed.loc["MTC", "linear_tpm_comparable"]
    assert keyed.loc["MTC", "benchmark_eligible"]
    assert keyed.loc["MTC", "availability_reason"] == "nonlinear_or_proxy_expression_scale"
    assert keyed.loc["RB", "linear_tpm_comparable"]
    assert not keyed.loc["RB", "benchmark_eligible"]
    assert keyed.loc["RB", "availability_reason"] == "no_pass_samples_high_concentration_source"
    assert expression.available_representative_cohorts(
        linear_tpm_comparable=True,
        benchmark_eligible=True,
    ) == ["PRAD"]

    mtc = expression.representative_cohort_samples(
        "MTC", format="long", include_provenance=True, sample_qc="artifact"
    )
    assert set(mtc["source_sample_qc_reasons"]) == {"nonlinear_or_proxy_expression_scale"}
    assert set(mtc["source_scale_class"]) == {"microarray_tpm_proxy"}
    assert not mtc["linear_tpm_comparable"].any()
    assert mtc.attrs["source_scale_class"] == "microarray_tpm_proxy"
    assert mtc.attrs["linear_tpm_comparable"] is False


def test_representatives_reject_mismatched_artifact_sample_qc(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1"],
            "Symbol": ["GENE1"],
            "PRAD__rep1": [1.0],
        }
    ).to_parquet(d / "PRAD.parquet", index=False)
    _write_artifact_build_metadata(tmp_path, "PRAD", sample_qc="all")

    with pytest.raises(ValueError, match="sample_qc mismatch"):
        expression.representative_cohort_samples("PRAD")

    audit = expression.representative_cohort_samples("PRAD", sample_qc="artifact")
    assert audit.attrs["sample_qc"] == "artifact"
    assert audit.attrs["artifact_sample_qc"] == "all"


def test_representative_empty_long_schema_includes_requested_provenance(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    shard_dir = tmp_path / "cancer-reference-expression-representatives"
    shard_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1"],
            "Symbol": ["GENE1"],
            "OTHER__rep1": [1.0],
        }
    ).to_parquet(shard_dir / "OTHER.parquet", index=False)

    df = expression.representative_cohort_samples("PRAD", format="long", include_provenance=True)

    assert df.empty
    assert list(df.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "cancer_code",
        "representative_id",
        "expression",
        "source_cohort",
        "source_version",
        "source_project",
        "source_diagnosis",
        "source_morphology",
        "source_sample",
        "source_group_id",
        "n_cohort_samples",
        "sample_qc",
        "sample_qc_requested",
        "source_sample_qc",
        "sample_qc_effective",
        "sample_qc_fallback_reason",
        "sample_qc_policy_version",
        "source_sample_qc_reasons",
        "n_qc_pass",
        "n_qc_warn",
        "n_qc_fail",
        "source_scale_class",
        "linear_tpm_comparable",
        "recommended_for_absolute_tpm_floor",
        "selection_scale_class",
        "representative_role",
        "benchmark_eligible",
        "partition_role",
        "partition_reason",
        "partition_policy_version",
        "partition_validation_target",
        "partition_status",
        "n_partition_train",
        "n_partition_validation",
        "n_partition_validation_external",
        "n_partition_audit_only",
        "review_source",
        "review_note",
        "selection_rank",
        "selection_method",
        "selection_basis",
        "artifact_schema_version",
        "data_version",
        "source_matrix_version",
    ]
    assert df.attrs["schema_version"] == expression.REPRESENTATIVE_ARTIFACT_SCHEMA_VERSION
    assert df.attrs["representative_id_style"] == "pirlygenes"


def test_representative_id_style_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    (tmp_path / "cancer-reference-expression-representatives").mkdir(parents=True)

    with pytest.raises(ValueError, match="representative_id_style"):
        expression.representative_cohort_samples("PRAD", representative_id_style="legacy")

    with pytest.raises(ValueError, match="gene_id_style"):
        expression.representative_cohort_samples("PRAD", gene_id_style="legacy")


def test_representatives_can_return_pirlygenes_legacy_gene_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000310560"],
            "Symbol": ["PAXX"],
            "PRAD__rep1": [7.0],
        }
    ).to_parquet(d / "PRAD.parquet", index=False)

    canonical = expression.representative_cohort_samples("PRAD")
    legacy = expression.representative_cohort_samples("PRAD", gene_id_style="pirlygenes")
    legacy_long = expression.representative_cohort_samples(
        "PRAD", format="long", gene_id_style="pirlygenes"
    )

    assert canonical.loc[0, "Ensembl_Gene_ID"] == "ENSG00000310560"
    assert canonical.attrs["gene_id_style"] == "oncoref"
    assert legacy.loc[0, "Ensembl_Gene_ID"] == "ENSG00000148362"
    assert legacy.loc[0, "Symbol"] == "PAXX"
    assert legacy.loc[0, "PRAD_rep01"] == pytest.approx(7.0)
    assert legacy.attrs["gene_id_style"] == "pirlygenes"
    assert legacy.attrs["gene_universe_delta_n"] == 264
    assert legacy_long.loc[0, "Ensembl_Gene_ID"] == "ENSG00000148362"
    assert legacy_long.attrs["gene_id_style"] == "pirlygenes"


def test_representatives_can_return_pirlygenes_sequence_identical_gene_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000206034"],
            "Symbol": ["DEFB109B"],
            "PRAD__rep1": [11.0],
        }
    ).to_parquet(d / "PRAD.parquet", index=False)

    canonical = expression.representative_cohort_samples("PRAD")
    legacy = expression.representative_cohort_samples("PRAD", gene_id_style="pirlygenes")
    legacy_long = expression.representative_cohort_samples(
        "PRAD", format="long", gene_id_style="pirlygenes"
    )

    assert canonical.loc[0, "Ensembl_Gene_ID"] == "ENSG00000206034"
    assert legacy.loc[0, "Ensembl_Gene_ID"] == "ENSG00000205989"
    assert legacy.loc[0, "Symbol"] == "DEFB109C"
    assert legacy.loc[0, "PRAD_rep01"] == pytest.approx(11.0)
    assert legacy_long.loc[0, "Ensembl_Gene_ID"] == "ENSG00000205989"
    assert legacy_long.loc[0, "Symbol"] == "DEFB109C"


def test_representative_wide_does_not_fragment_genes(monkeypatch, tmp_path):
    # The #465-class bug: cohorts quantified on different Ensembl releases carry the SAME
    # locus under different alias symbols (and one as the raw id). The wide combiner must
    # key on the canonical gene id, so the gene stays ONE row covered by all cohorts —
    # never three mutually-disjoint sparse alias rows.
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG234", "ENSG999"],
            "Symbol": ["AP000959.1", "PRAME"],
            "A__rep1": [1.0, 2.0],
        }
    ).to_parquet(d / "A.parquet", index=False)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG234", "ENSG888"],
            "Symbol": ["RP11-844P9.5", "MAGEA4"],
            "B__rep1": [3.0, 4.0],
        }
    ).to_parquet(d / "B.parquet", index=False)
    pd.DataFrame(
        {"Ensembl_Gene_ID": ["ENSG234"], "Symbol": ["ENSG234"], "C__rep1": [5.0]}  # raw-id backfill
    ).to_parquet(d / "C.parquet", index=False)

    w = expression.representative_cohort_samples(format="wide")
    assert len(w) == w["Ensembl_Gene_ID"].nunique()  # one row per gene id (no fragmentation)
    shared = w[w["Ensembl_Gene_ID"] == "ENSG234"]
    assert len(shared) == 1
    assert shared["Symbol"].iloc[0] != "ENSG234"  # canonical symbol prefers a real alias
    # all three cohorts' values land on that single row
    assert bool(shared[["A_rep01", "B_rep01", "C_rep01"]].notna().all(axis=1).iloc[0])

    internal = expression.representative_cohort_samples(
        format="wide", representative_id_style="internal"
    )
    assert {"A__rep1", "B__rep1", "C__rep1"} <= set(internal.columns)

    # the long form carries the same canonical symbol (so a downstream pivot won't fragment)
    lg = expression.representative_cohort_samples(format="long")
    assert lg.loc[lg["Ensembl_Gene_ID"] == "ENSG234", "Symbol"].nunique() == 1


def test_representative_wide_merges_alt_haplotype_aliases(monkeypatch, tmp_path):
    # An alt-haplotype/archived id in one cohort must collapse onto its primary-contig id
    # via the shipped ensembl-id-aliases migration map (not just unversioning) when another
    # cohort carries the primary — so the two cohorts land on ONE canonical row.
    from oncoref.gene_ids import ensembl_id_aliases

    alt, primary = next((a, p) for a, p in ensembl_id_aliases().items() if a != p)
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    pd.DataFrame({"Ensembl_Gene_ID": [alt], "Symbol": ["X"], "A__rep1": [1.0]}).to_parquet(
        d / "A.parquet", index=False
    )
    pd.DataFrame({"Ensembl_Gene_ID": [primary], "Symbol": ["X"], "B__rep1": [2.0]}).to_parquet(
        d / "B.parquet", index=False
    )
    w = expression.representative_cohort_samples(format="wide")
    assert list(w["Ensembl_Gene_ID"]) == [primary]  # one row, the canonical primary
    assert alt not in set(w["Ensembl_Gene_ID"])
    assert bool(w[["A_rep01", "B_rep01"]].notna().all(axis=1).iloc[0])  # both cohorts on it


def test_representative_wide_sums_alt_haplotype_within_cohort(monkeypatch, tmp_path):
    # When one cohort carries BOTH an alt-haplotype id and its primary (a full-assembly
    # quantification annotates the gene on both contigs), their per-sample TPMs must be
    # SUMMED under the canonical id (reads multi-map between copies), not deduped to one.
    from oncoref.gene_ids import ensembl_id_aliases

    alt, primary = next((a, p) for a, p in ensembl_id_aliases().items() if a != p)
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    # one cohort, one representative, the gene present under both ids
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": [primary, alt],
            "Symbol": ["G", "G-alt"],
            "A__rep1": [10.0, 3.0],
        }
    ).to_parquet(d / "A.parquet", index=False)

    w = expression.representative_cohort_samples(format="wide")
    assert list(w["Ensembl_Gene_ID"]) == [primary]  # collapsed onto the canonical id
    assert w["A_rep01"].iloc[0] == pytest.approx(13.0)  # 10 + 3 summed, not 10 or 3


def test_representative_log1p_sums_in_linear_space(monkeypatch, tmp_path):
    # The alt-haplotype sum must be taken in LINEAR TPM, then log1p applied — never
    # sum log-space values: log1p(10)+log1p(3) != log1p(10+3).
    from oncoref.gene_ids import ensembl_id_aliases

    alt, primary = next((a, p) for a, p in ensembl_id_aliases().items() if a != p)
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    pd.DataFrame(
        {"Ensembl_Gene_ID": [primary, alt], "Symbol": ["G", "G-alt"], "A__rep1": [10.0, 3.0]}
    ).to_parquet(d / "A.parquet", index=False)

    w = expression.representative_cohort_samples(format="wide", normalize="tpm_clean_log1p")
    assert list(w["Ensembl_Gene_ID"]) == [primary]
    assert w["A_rep01"].iloc[0] == pytest.approx(np.log1p(13.0))  # log1p(10+3)
    assert w["A_rep01"].iloc[0] != pytest.approx(np.log1p(10.0) + np.log1p(3.0))  # NOT Σlog1p


def _raw_matrix(tmp_path):
    # A tiny raw-TPM per-sample matrix (genes x samples) whose columns sum near 1e6.
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1", "ENSG2", "ENSG3"],
            "Symbol": ["A", "B", "C"],
            "s1": [500000.0, 300000.0, 200000.0],
            "s2": [100000.0, 600000.0, 300000.0],
        }
    )
    path = tmp_path / "PRAD.parquet"
    df.to_parquet(path, index=False)
    return path


def test_per_sample_matrix_cache_size_is_tunable():
    # The LRU is wired to the (env-configurable) constant, not a hardcoded literal,
    # so heavy pooling workflows can keep >2 matrices warm via CANCERDATA_PER_SAMPLE_CACHE.
    assert (
        expression._load_per_sample_matrix.cache_info().maxsize == expression._PER_SAMPLE_CACHE_SIZE
    )
    assert expression._PER_SAMPLE_CACHE_SIZE >= 1


def test_per_sample_cache_size_tolerates_malformed_env(monkeypatch):
    # A tuning knob must never break `import oncoref`: a malformed/empty value
    # falls back to the default rather than raising at parse time.
    for bad in ("", "abc", "2.5"):
        monkeypatch.setenv("CANCERDATA_PER_SAMPLE_CACHE", bad)
        assert expression._per_sample_cache_size() == 2
    monkeypatch.setenv("CANCERDATA_PER_SAMPLE_CACHE", "0")  # clamps to >=1
    assert expression._per_sample_cache_size() == 1
    monkeypatch.setenv("CANCERDATA_PER_SAMPLE_CACHE", "8")  # honored
    assert expression._per_sample_cache_size() == 8


def test_per_sample_expression_normalize_modes(tmp_path, monkeypatch):
    path = _raw_matrix(tmp_path)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)

    raw = expression.per_sample_expression("PRAD", normalize="tpm_raw")
    assert list(raw.columns) == ["Ensembl_Gene_ID", "Symbol", "s1", "s2"]
    assert raw["s1"].sum() == pytest.approx(1e6)

    clean = expression.per_sample_expression("PRAD", normalize="tpm_clean")
    # No technical/censored genes in this fixture -> the technical compartment is
    # empty and the biological compartment fills its 750k budget (clean_tpm's
    # two-compartment contract). A real matrix with technical genes sums to ~1e6.
    assert clean["s1"].sum() == pytest.approx(750000.0, rel=1e-6)
    assert list(clean.columns) == ["Ensembl_Gene_ID", "Symbol", "s1", "s2"]

    logged = expression.per_sample_expression("PRAD", normalize="tpm_clean_log1p")
    assert np.allclose(logged["s1"].to_numpy(), np.log1p(clean["s1"].to_numpy()))

    # hk mode runs (no HK panel genes in this fixture -> a no-op rescale, but valid).
    hk = expression.per_sample_expression("PRAD", normalize="tpm_clean_hk")
    assert list(hk.columns) == ["Ensembl_Gene_ID", "Symbol", "s1", "s2"]


def test_per_sample_expression_canonicalizes_alias_genes(tmp_path, monkeypatch):
    # per_sample_expression returns the DENSE canonical space (oncoref#135 item 6): an
    # alt-haplotype copy is summed into its primary in LINEAR TPM, a versioned id is
    # unversioned, a retired id is relabeled to its successor — every transform applied
    # AFTER the linear sum.
    from oncoref.gene_ids import ensembl_id_aliases

    alt, primary = next((a, p) for a, p in ensembl_id_aliases().items() if a != p)
    raw = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [primary, alt, "ENSG00000005955", "ENSG00000141510.5"],
            "Symbol": ["G", "G-alt", "GGNBP2-old", "TP53"],
            "s1": [100000.0, 30000.0, 1000.0, 200000.0],
            "s2": [50000.0, 10000.0, 2000.0, 400000.0],
        }
    )
    path = tmp_path / "X.parquet"
    raw.to_parquet(path, index=False)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)

    rawout = expression.per_sample_expression("X", normalize="tpm_raw").set_index("Ensembl_Gene_ID")
    assert alt not in rawout.index  # alt-haplotype copy folded into its primary
    assert rawout.loc[primary, "s1"] == pytest.approx(130000.0)  # 100000 + 30000, linear
    assert "ENSG00000141510" in rawout.index  # versioned id -> unversioned canonical
    assert "ENSG00000278311" in rawout.index  # GGNBP2 retired id -> successor
    assert "ENSG00000005955" not in rawout.index

    # log1p is applied AFTER the linear alt-copy sum, never log1p(a)+log1p(b)
    clean = expression.per_sample_expression("X", normalize="tpm_clean").set_index(
        "Ensembl_Gene_ID"
    )
    logged = expression.per_sample_expression("X", normalize="tpm_clean_log1p").set_index(
        "Ensembl_Gene_ID"
    )
    assert np.allclose(
        logged.loc[primary, ["s1", "s2"]].to_numpy(dtype=float),
        np.log1p(clean.loc[primary, ["s1", "s2"]].to_numpy(dtype=float)),
    )


def _mock_pan_cancer_data_without_computed_aggregates(monkeypatch, frame):
    monkeypatch.setattr(expression, "get_data", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(
        expression, "_computed_expression_reference_members", lambda cancer_code: ()
    )


def test_pan_cancer_expression_canonicalizes_alias_genes(monkeypatch):
    # pan_cancer_expression is dense canonical too: cohort/tissue columns are linear
    # abundances (nTPM/FPKM), so summing an alt copy into its primary row is exact.
    from oncoref.gene_ids import ensembl_id_aliases

    alt, primary = next((a, p) for a, p in ensembl_id_aliases().items() if a != p)
    fixture = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [primary, alt],
            "Symbol": ["G", "G-alt"],
            "nTPM_liver": [3.0, 7.0],
        }
    )
    _mock_pan_cancer_data_without_computed_aggregates(monkeypatch, fixture)
    out = expression.pan_cancer_expression(normalize=None).set_index("Ensembl_Gene_ID")
    assert list(out.index) == [primary]  # alt folded into primary
    assert out.loc[primary, "liver_nTPM_raw"] == pytest.approx(10.0)  # 3 + 7 summed


def test_housekeeping_normalize_divides_by_panel_size_factor(monkeypatch):
    import oncoref.gene_families as gf
    import oncoref.normalization as norm

    monkeypatch.setattr(
        gf, "clean_tpm_biological_housekeeping_gene_ids", lambda: frozenset({"ENSG_HK"})
    )
    monkeypatch.setattr(
        norm,
        "housekeeping_reference_profile",
        lambda: pd.DataFrame({"Ensembl_Gene_ID": ["ENSG_HK"], "reference_tpm": [100.0]}),
    )
    df = pd.DataFrame(
        {"Ensembl_Gene_ID": ["ENSG_HK", "ENSG_X"], "Symbol": ["HK", "X"], "s1": [100.0, 50.0]}
    )
    out = expression._housekeeping_normalize(df, ["s1"])
    # Each column divided by its housekeeping size factor -> gene/HK ratio is preserved.
    assert out.loc[1, "s1"] / out.loc[0, "s1"] == pytest.approx(0.5)


def test_housekeeping_normalize_blanks_sparse_housekeeping_denominator(monkeypatch):
    import oncoref.gene_families as gf
    import oncoref.normalization as norm

    panel = frozenset({"HK1", "HK2", "HK3"})
    monkeypatch.setattr(gf, "clean_tpm_biological_housekeeping_gene_ids", lambda: panel)
    monkeypatch.setattr(
        norm,
        "housekeeping_reference_profile",
        lambda: pd.DataFrame(
            {"Ensembl_Gene_ID": ["HK1", "HK2", "HK3"], "reference_tpm": [100.0, 100.0, 100.0]}
        ),
    )
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["HK1", "HK2", "HK3", "ENSG_X"],
            "Symbol": ["HK1", "HK2", "HK3", "X"],
            "good": [100.0, 120.0, 80.0, 50.0],
            "sparse": [0.0, 0.0, 100.0, 50.0],
        }
    )

    with pytest.warns(RuntimeWarning, match="housekeeping normalization skipped"):
        out = expression._housekeeping_normalize(df, ["good", "sparse"])

    assert out.loc[out["Symbol"] == "X", "good"].iloc[0] > 0
    assert out["sparse"].isna().all()


def test_sample_expression_qc_flags_sparse_samples(tmp_path, monkeypatch):
    import oncoref.gene_families as gf
    from oncoref import expression_registry

    panel = frozenset({"HK1", "HK2", "HK3"})
    monkeypatch.setattr(gf, "clean_tpm_biological_housekeeping_gene_ids", lambda: panel)
    raw = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["HK1", "HK2", "HK3", "ENSG_A", "ENSG_B", "ENSG_C"],
            "Symbol": ["HK1", "HK2", "HK3", "A", "B", "C"],
            "good": [100.0, 120.0, 80.0, 100.0, 100.0, 100.0],
            "sparse": [0.0, 0.0, 20.0, 1000.0, 0.0, 0.0],
        }
    )
    path = tmp_path / "X.parquet"
    raw.to_parquet(path, index=False)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)
    monkeypatch.setattr(
        expression.source_matrices,
        "cohort_info",
        lambda code: {"source_cohort": "TEST_SOURCE", "n_samples": 2},
    )
    monkeypatch.setattr(
        expression_registry,
        "sources_for_cancer_code",
        lambda code: [
            expression_registry.ExpressionSource(
                id="test-source",
                category="expression",
                cancer_codes=("X",),
                source_type="gdc",
                unit="TPM",
                source_cohort="TEST_SOURCE",
            )
        ],
    )

    qc = expression.sample_expression_qc(
        "X",
        min_detected_genes=3,
        min_housekeeping_detected=2,
        max_top_gene_fraction=0.8,
    ).set_index("sample_id")

    assert bool(qc.loc["good", "passes_expression_qc"]) is True
    assert qc.loc["good", "sample_qc_status"] == "pass"
    assert qc.loc["good", "source_scale_class"] == "linear_rnaseq_tpm"
    assert bool(qc.loc["good", "linear_tpm_comparable"]) is True
    assert qc.loc["good", "housekeeping_genes_detected"] == 3
    assert bool(qc.loc["sparse", "passes_expression_qc"]) is False
    assert qc.loc["sparse", "sample_qc_status"] == "fail"
    assert qc.loc["sparse", "n_detected_raw"] == 2
    assert qc.loc["sparse", "n_detected_clean_biological"] == 2
    assert qc.loc["sparse", "n_detected_genes"] == 2
    assert qc.loc["sparse", "housekeeping_genes_detected"] == 1
    assert "low_housekeeping_detection" in qc.loc["sparse", "sample_qc_reasons"]
    assert "high_top_gene_fraction" in qc.loc["sparse", "sample_qc_reasons"]
    assert "qc_flags" not in qc.columns
    assert "qc_status" not in qc.columns
    assert "qc_reasons" not in qc.columns
    assert "top1_fraction_raw" not in qc.columns
    assert "top10_fraction_raw" not in qc.columns


def test_sample_expression_qc_flags_high_literal_zero_fraction():
    raw = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [f"ENSG{i:011d}" for i in range(10)],
            "Symbol": [f"G{i}" for i in range(10)],
            "sparse_zero": [1.0, 1.0, *([0.0] * 8)],
            "less_sparse": [1.0, 1.0, 1.0, 1.0, *([0.0] * 6)],
        }
    )

    qc = expression.sample_expression_qc_from_matrix(
        raw,
        cancer_type="X",
        source_metadata={
            "source_type": "geo",
            "unit": "TPM",
            "source_scale_class": "linear_rnaseq_tpm",
            "linear_tpm_comparable": True,
        },
        min_detected_genes=1,
        min_housekeeping_detected=0,
        max_zero_fraction=0.7,
        max_top_gene_fraction=1.0,
        max_top10_gene_fraction=1.0,
    ).set_index("sample_id")

    assert qc.loc["sparse_zero", "zero_fraction_raw"] == pytest.approx(0.8)
    assert qc.loc["sparse_zero", "sample_qc_status"] == "fail"
    assert "high_zero_fraction" in qc.loc["sparse_zero", "sample_qc_reasons"]
    assert qc.loc["less_sparse", "zero_fraction_raw"] == pytest.approx(0.6)
    assert qc.loc["less_sparse", "sample_qc_status"] == "pass"


def test_sample_expression_qc_proxy_scale_warns_without_rnaseq_fail_gates():
    raw = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [f"ENSG{i:011d}" for i in range(20)],
            "Symbol": [f"G{i}" for i in range(20)],
            "proxy": [1000.0, *([0.0] * 19)],
        }
    )

    qc = expression.sample_expression_qc_from_matrix(
        raw,
        cancer_type="X",
        source_metadata={
            "source_type": "microarray",
            "unit": "TPM proxy",
            "source_scale_class": "microarray_tpm_proxy",
            "linear_tpm_comparable": False,
            "tpm_proxy": True,
        },
        min_detected_genes=5000,
        min_housekeeping_detected=10,
        max_zero_fraction=0.1,
        max_top_gene_fraction=0.1,
        max_top10_gene_fraction=0.1,
    ).set_index("sample_id")

    assert qc.loc["proxy", "sample_qc_status"] == "warn"
    assert qc.loc["proxy", "passes_expression_qc"]
    assert qc.loc["proxy", "sample_qc_reasons"] == "nonlinear_or_proxy_expression_scale"
    assert not qc.loc["proxy", "recommended_for_absolute_tpm_floor"]


def test_sample_expression_qc_records_configurable_housekeeping_floor():
    raw = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000188612", "ENSG00000143761", "ENSG00000141510"],
            "Symbol": ["SUMO2", "ARF1", "TP53"],
            "sample": [25.0, 75.0, 2.0],
        }
    )

    qc = expression.sample_expression_qc_from_matrix(
        raw,
        cancer_type="X",
        source_metadata={
            "source_scale_class": "linear_rnaseq_tpm",
            "linear_tpm_comparable": True,
        },
        min_detected_genes=1,
        min_housekeeping_detected=1,
        housekeeping_detection_floor_tpm=50.0,
        min_housekeeping_fraction_above_floor=0.75,
        max_zero_fraction=1.0,
        max_top_gene_fraction=1.0,
        max_top10_gene_fraction=1.0,
    ).set_index("sample_id")

    assert qc.loc["sample", "housekeeping_detection_floor_tpm"] == 50.0
    assert qc.loc["sample", "housekeeping_genes_above_floor"] == 1
    assert qc.loc["sample", "housekeeping_fraction_above_floor"] == pytest.approx(0.5)
    assert qc.loc["sample", "sample_qc_status"] == "fail"
    assert "low_housekeeping_floor_fraction" in qc.loc["sample", "sample_qc_reasons"]


def test_housekeeping_cancer_expression_coverage_from_matrix_reports_low_tail_stats():
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG_HK1", "ENSG_HK2", "ENSG_OTHER"],
            "Symbol": ["HK1", "HK2", "OTHER"],
            "s1": [40.0, 0.0, 10.0],
            "s2": [20.0, 50.0, 20.0],
        }
    )

    out = expression.housekeeping_cancer_expression_coverage_from_matrix(
        df,
        cancer_type="LUAD",
        source_metadata={
            "source_cohort": "TEST_SOURCE",
            "source_type": "test",
            "unit": "TPM_clean",
            "source_scale_class": "linear_rnaseq_tpm",
            "linear_tpm_comparable": True,
        },
        panel_ids=["ENSG_HK1", "ENSG_HK2", "ENSG_MISSING"],
        housekeeping_detection_floor_tpm=30.0,
        sample_qc="provided",
    )

    keyed = out.set_index("Ensembl_Gene_ID")
    assert list(keyed.index) == ["ENSG_HK1", "ENSG_HK2", "ENSG_MISSING"]
    assert keyed.loc["ENSG_HK1", "panel_member_present"]
    assert keyed.loc["ENSG_HK1", "n_samples"] == 2
    assert keyed.loc["ENSG_HK1", "n_detected_samples"] == 2
    assert keyed.loc["ENSG_HK1", "n_above_floor_samples"] == 1
    assert keyed.loc["ENSG_HK1", "fraction_above_floor"] == pytest.approx(0.5)
    assert keyed.loc["ENSG_HK1", "min_tpm"] == pytest.approx(20.0)
    assert keyed.loc["ENSG_HK1", "median_tpm"] == pytest.approx(30.0)
    assert not keyed.loc["ENSG_HK1", "passes_p5_floor"]

    assert keyed.loc["ENSG_HK2", "n_detected_samples"] == 1
    assert not keyed.loc["ENSG_MISSING", "panel_member_present"]
    assert keyed.loc["ENSG_MISSING", "n_measured_samples"] == 0
    assert keyed.loc["ENSG_MISSING", "fraction_above_floor"] == pytest.approx(0.0)
    assert pd.isna(keyed.loc["ENSG_MISSING", "p5_tpm"])
    assert keyed["recommended_for_absolute_tpm_floor"].all()
    assert out.attrs["issue"] == "#202"


def test_housekeeping_cancer_expression_coverage_threads_sample_qc_and_source_scale(monkeypatch):
    matrix = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG_HK"],
            "Symbol": ["HK"],
            "sample": [100.0],
        }
    )

    def fake_per_sample_expression(code, *, normalize, sample_qc, auto_fetch):
        assert normalize == "tpm_clean"
        assert sample_qc == "pass"
        assert auto_fetch is False
        return matrix.copy()

    def fake_source_metadata(code):
        return {
            "source_cohort": f"{code}_SOURCE",
            "source_type": "microarray" if code == "MTC" else "bulk RNA-seq",
            "unit": "TPM proxy" if code == "MTC" else "TPM",
            "source_scale_class": "microarray_tpm_proxy" if code == "MTC" else "linear_rnaseq_tpm",
            "linear_tpm_comparable": code != "MTC",
        }

    monkeypatch.setattr(expression, "per_sample_expression", fake_per_sample_expression)
    monkeypatch.setattr(expression, "_selected_expression_source_metadata", fake_source_metadata)

    out = expression.housekeeping_cancer_expression_coverage(
        ["LUAD", "MTC"],
        sample_qc="pass",
        auto_fetch=False,
        panel_ids=["ENSG_HK"],
        housekeeping_detection_floor_tpm=30.0,
    )

    keyed = out.set_index("cancer_code")
    assert set(keyed.index) == {"LUAD", "MTC"}
    assert keyed.loc["LUAD", "recommended_for_absolute_tpm_floor"]
    assert not keyed.loc["MTC", "recommended_for_absolute_tpm_floor"]
    assert keyed.loc["MTC", "source_scale_class"] == "microarray_tpm_proxy"
    assert set(out["sample_qc"]) == {"pass"}
    assert set(out["expression_space"]) == {"tpm_clean"}


def test_per_sample_expression_filters_by_sample_qc(tmp_path, monkeypatch):
    raw = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1", "ENSG2"],
            "Symbol": ["A", "B"],
            "pass_sample": [10.0, 20.0],
            "warn_sample": [30.0, 40.0],
            "fail_sample": [50.0, 60.0],
        }
    )
    path = tmp_path / "X.parquet"
    raw.to_parquet(path, index=False)
    expression._load_per_sample_matrix.cache_clear()
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)
    monkeypatch.setattr(
        expression,
        "sample_expression_qc",
        lambda code, **k: pd.DataFrame(
            {
                "sample_id": ["pass_sample", "warn_sample", "fail_sample"],
                "sample_qc_status": ["pass", "warn", "fail"],
            }
        ),
    )

    passed = expression.per_sample_expression("X", normalize="tpm_raw", sample_qc="pass")
    assert expression.sample_columns(passed) == ["pass_sample"]

    pass_or_warn = expression.per_sample_expression(
        "X", normalize="tpm_raw", sample_qc="pass_or_warn"
    )
    assert expression.sample_columns(pass_or_warn) == ["pass_sample", "warn_sample"]

    all_samples = expression.per_sample_expression("X", normalize="tpm_raw", sample_qc="all")
    assert expression.sample_columns(all_samples) == ["pass_sample", "warn_sample", "fail_sample"]


def test_source_matrix_sample_qc_manifest_missing_is_schema_stable(tmp_path, monkeypatch):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))

    out = expression.source_matrix_sample_qc_manifest(auto_fetch=False)

    assert out.empty
    assert "sample_qc_status" in out.columns
    assert "schema_version" not in out.attrs
    assert out.attrs["data_version"] == expression.DATA_VERSION
    assert "missing_reason" in out.attrs
    with pytest.raises(FileNotFoundError, match="source-matrix-sample-qc"):
        expression.source_matrix_sample_qc_manifest(auto_fetch=False, on_missing="raise")


def test_source_matrix_sample_qc_manifest_reads_filters_and_exports(tmp_path, monkeypatch):
    import oncoref

    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    pd.DataFrame(
        {
            "cancer_code": ["PRAD", "PRAD", "BRCA"],
            "sample_id": ["pass_sample", "warn_sample", "fail_sample"],
            "sample_qc_status": ["pass", "warn", "fail"],
            "source_cohort": ["SRC", "SRC", "SRC2"],
            "custom_future_column": [1, 2, 3],
        }
    ).to_csv(tmp_path / expression.SOURCE_MATRIX_SAMPLE_QC_MANIFEST_PATH, index=False)

    out = expression.source_matrix_sample_qc_manifest(
        "prostate", sample_qc="pass_or_warn", auto_fetch=False
    )

    assert out["sample_id"].tolist() == ["pass_sample", "warn_sample"]
    assert out["custom_future_column"].tolist() == [1, 2]
    assert out.attrs["path"].endswith(expression.SOURCE_MATRIX_SAMPLE_QC_MANIFEST_PATH)
    assert oncoref.source_matrix_sample_qc_manifest is expression.source_matrix_sample_qc_manifest
    assert not hasattr(oncoref, "SOURCE_MATRIX_SAMPLE_QC_MANIFEST_SCHEMA_VERSION")


def test_expression_artifact_build_metadata_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    monkeypatch.setattr(
        expression.source_matrices,
        "registry",
        lambda: pd.DataFrame(
            {
                "cancer_code": ["PRAD", "BRCA"],
                "source_cohort": ["CANONICAL_PRAD", "CANONICAL_BRCA"],
            }
        ),
    )
    pd.DataFrame(
        {
            "cancer_code": ["PRAD", "BRCA"],
            "source_cohort": ["SRC_PRAD", "SRC_BRCA"],
            "sample_qc": ["pass", "all"],
            "n_source_samples": [5, 7],
            "n_cohort_samples": [4, 7],
            "extra_future_column": ["x", "y"],
        }
    ).to_csv(tmp_path / expression.EXPRESSION_ARTIFACT_BUILD_METADATA_PATH, index=False)
    (tmp_path / expression.EXPRESSION_ARTIFACT_BUILD_METADATA_JSON_PATH).write_text(
        json.dumps(
            {
                "artifact": "expression-derived-shards",
                "sample_qc": "pass",
                "n_cohorts": 2,
            }
        )
        + "\n"
    )

    meta = expression.expression_artifact_build_metadata("PRAD", auto_fetch=False)
    summary = expression.expression_artifact_build_summary(auto_fetch=False)

    assert meta["source_cohort"].tolist() == ["CANONICAL_PRAD"]
    assert meta["build_source_cohort"].tolist() == ["SRC_PRAD"]
    assert meta["extra_future_column"].tolist() == ["x"]
    assert "n_qc_pass" in meta.columns
    assert (
        meta.attrs["schema_version"] == expression.EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION
    )
    assert summary["artifact"] == "expression-derived-shards"
    assert summary["schema_version"] == expression.EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION
    assert summary["path"].endswith(expression.EXPRESSION_ARTIFACT_BUILD_METADATA_JSON_PATH)


def test_expression_artifact_build_metadata_missing_can_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))

    out = expression.expression_artifact_build_metadata(auto_fetch=False)
    summary = expression.expression_artifact_build_summary(auto_fetch=False)

    assert out.empty
    assert (
        out.attrs["schema_version"] == expression.EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION
    )
    assert summary["missing_reason"]
    with pytest.raises(FileNotFoundError, match=r"expression-artifact-build-metadata\.csv"):
        expression.expression_artifact_build_metadata(auto_fetch=False, on_missing="raise")
    with pytest.raises(FileNotFoundError, match=r"expression-artifact-build-metadata\.json"):
        expression.expression_artifact_build_summary(auto_fetch=False, on_missing="raise")


def test_per_sample_expression_gene_and_proteoform_levels(tmp_path, monkeypatch):
    # ENSG1+ENSG2 are an identical-protein group; per_sample_expression(proteoform=True)
    # sums them per sample. Gene-level and proteoform-level are both available.
    import oncoref.proteoforms as pmod

    path = _raw_matrix(tmp_path)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)
    monkeypatch.setattr(
        pmod, "proteoform_group_map", lambda *, scope="cta": {"A/B": ("ENSG1", "ENSG2")}
    )

    gene = expression.per_sample_expression("PRAD", normalize="tpm_raw")
    assert pmod.expression_level(gene) == "gene" and len(gene) == 3

    pf = expression.per_sample_expression("PRAD", normalize="tpm_raw", proteoform=True)
    assert pmod.expression_level(pf) == "proteoform" and len(pf) == 2  # group merged
    # A/B summed per sample: s1 = 500000 + 300000 = 800000
    a_row = pf[pf["proteoform_key"] == "A/B"]
    assert a_row["s1"].iloc[0] == pytest.approx(800000.0)


def test_per_sample_expression_proteoform_log_sums_in_linear_space(tmp_path, monkeypatch):
    # log1p + proteoform must sum members in LINEAR TPM then log1p — NOT sum the logs.
    import oncoref.proteoforms as pmod

    path = _raw_matrix(tmp_path)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)
    monkeypatch.setattr(
        pmod, "proteoform_group_map", lambda *, scope="cta": {"A/B": ("ENSG1", "ENSG2")}
    )
    lin = expression.per_sample_expression("PRAD", normalize="tpm_clean", proteoform=True)
    log = expression.per_sample_expression("PRAD", normalize="tpm_clean_log1p", proteoform=True)
    assert np.allclose(log["s1"].to_numpy(), np.log1p(lin["s1"].to_numpy(dtype=float)))


def test_per_sample_expression_memoizes_and_copies(tmp_path, monkeypatch):
    path = _raw_matrix(tmp_path)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)
    expression._load_per_sample_matrix.cache_clear()

    a = expression.per_sample_expression("PRAD", normalize="tpm_clean")
    expression.per_sample_expression("PRAD", normalize="tpm_clean")  # served from cache
    info = expression._load_per_sample_matrix.cache_info()
    assert info.misses == 1 and info.hits >= 1

    # Each call returns an independent copy — mutating one must not corrupt the cache.
    col = a.columns[2]
    a.loc[0, col] = -12345.0
    c = expression.per_sample_expression("PRAD", normalize="tpm_clean")
    assert c.loc[0, col] != -12345.0


def test_per_sample_expression_bad_normalize(tmp_path, monkeypatch):
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: _raw_matrix(tmp_path))
    with pytest.raises(ValueError, match="normalize must be one of"):
        expression.per_sample_expression("PRAD", normalize="zscore")


def test_per_sample_expression_no_autofetch_raises(monkeypatch, tmp_path):
    missing = tmp_path / "nope.parquet"
    monkeypatch.setattr(expression.source_matrices, "local_path", lambda code: missing)
    with pytest.raises(FileNotFoundError, match="not cached"):
        expression.per_sample_expression("PRAD", auto_fetch=False)


def test_cohort_mean_expression(monkeypatch):
    fixture = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E2"],
            "Symbol": ["A", "B"],
            "s1": [10.0, 0.0],
            "s2": [20.0, 4.0],
            "s3": [30.0, 2.0],
        }
    )
    monkeypatch.setattr(expression, "per_sample_expression", lambda *a, **k: fixture.copy())
    mean = expression.cohort_mean_expression("X", statistic="mean")
    assert list(mean.columns) == ["Ensembl_Gene_ID", "Symbol", "expression"]
    assert dict(zip(mean["Symbol"], mean["expression"])) == {"A": 20.0, "B": 2.0}
    median = expression.cohort_mean_expression("X", statistic="median")
    assert dict(zip(median["Symbol"], median["expression"])) == {"A": 20.0, "B": 2.0}


def test_cohort_mean_expression_threads_proteoform_and_scope(monkeypatch):
    # cohort_mean delegates the collapse to per_sample_expression — it must pass
    # proteoform= and scope= through.
    from oncoref import expression_level

    seen = {}

    def fake_per_sample(code, *, normalize, auto_fetch, proteoform, scope, sample_qc):
        seen.update(proteoform=proteoform, scope=scope, sample_qc=sample_qc)
        return pd.DataFrame(
            {
                "proteoform_key": ["E1"],
                "Ensembl_Gene_ID": ["E1"],
                "Symbol": ["A"],
                "proteoform_members": ["A"],
                "s1": [1.0],
                "s2": [3.0],
            }
        )

    monkeypatch.setattr(expression, "per_sample_expression", fake_per_sample)
    out = expression.cohort_mean_expression("X", proteoform=True, scope="genome")
    assert seen == {"proteoform": True, "scope": "genome", "sample_qc": "pass"}
    assert expression_level(out) == "proteoform"  # proteoform_key carried through


def test_cancer_reference_expression_long_and_wide(monkeypatch):
    pct = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E2"],
            "Symbol": ["A", "B"],
            "p25": [1.0, 2.0],
            "p50": [3.0, 4.0],
            "p75": [5.0, 6.0],
        }
    )
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X", "Y"])
    _mock_missing_artifact_build_metadata(monkeypatch)
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(expression, "cohort_gene_percentiles", lambda *a, **k: pct.copy())

    long = expression.cancer_reference_expression("x", genes=["E1"])
    assert list(long.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "Proteoform_ID",
        "Member_Ensembl_Gene_IDs",
        "cancer_code",
        "normalization",
        "source_cohort",
        "source_project",
        "source_version",
        "source_type",
        "source_unit",
        "source_scale_class",
        "linear_tpm_comparable",
        "tumor_origin",
        "metastasis_site",
        "n_reference_genes",
        "n_reference_samples",
        "n_samples",
        "n_detected",
        "processing_pipeline",
        "notes",
        "reference_method",
        "sample_qc",
        "sample_qc_effective",
        "data_version",
        "source_matrix_version",
        "expression",
        "q1",
        "q3",
    ]
    assert long["cancer_code"].tolist() == ["X"]
    assert long["normalization"].tolist() == ["tpm_clean"]
    assert long["reference_method"].tolist() == ["percentile_shard"]
    assert long["sample_qc"].tolist() == ["pass"]
    assert long["expression"].tolist() == [3.0]
    assert long["q1"].tolist() == [1.0] and long["q3"].tolist() == [5.0]
    assert expression.cancer_reference_expression("x", genes=["E1"], normalize="clean_tpm").equals(
        long
    )

    wide = expression.cancer_reference_expression(["x", "y"], format="wide")
    assert {"X_TPM_clean", "Y_TPM_clean"} <= set(wide.columns)
    assert wide.loc[wide["Symbol"] == "A", "X_TPM_clean"].iloc[0] == 3.0


def test_cancer_reference_expression_can_return_pirlygenes_legacy_gene_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    shard_dir = tmp_path / "cancer-reference-expression-percentiles"
    shard_dir.mkdir(parents=True)
    row = {
        "Ensembl_Gene_ID": "ENSG00000310560",
        "Symbol": "PAXX",
    }
    for bp in sorted({*_BREAKPOINTS, 25, 75}):
        row[f"p{bp}"] = np.log1p(float(bp))
    pd.DataFrame([row]).to_parquet(shard_dir / "PRAD.parquet", index=False)

    canonical = expression.cancer_reference_expression(
        "PRAD",
        genes="ENSG00000148362",
    )
    legacy = expression.cancer_reference_expression(
        "PRAD",
        genes="ENSG00000148362",
        gene_id_style="pirlygenes",
    )
    wide = expression.cancer_reference_expression(
        "PRAD",
        genes="PAXX",
        format="wide",
        gene_id_style="pirlygenes",
    )

    assert canonical.loc[0, "Ensembl_Gene_ID"] == "ENSG00000310560"
    assert canonical.attrs["gene_id_style"] == "oncoref"
    assert legacy.loc[0, "Ensembl_Gene_ID"] == "ENSG00000148362"
    assert legacy.loc[0, "Symbol"] == "PAXX"
    assert legacy.loc[0, "expression"] == pytest.approx(canonical.loc[0, "expression"])
    assert legacy.attrs["gene_id_style"] == "pirlygenes"
    assert legacy.attrs["gene_universe_delta_n"] == 10
    assert wide.loc[0, "Ensembl_Gene_ID"] == "ENSG00000148362"
    assert wide.attrs["gene_id_style"] == "pirlygenes"


def test_cancer_reference_expression_bad_gene_id_style():
    with pytest.raises(ValueError, match="gene_id_style"):
        expression.cancer_reference_expression("PRAD", gene_id_style="legacy")


def test_cancer_reference_expression_adds_proteoform_bridge_without_collapse(monkeypatch):
    pct = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [
                "ENSG00000154545",  # MAGED4 cDNA-identical group member
                "ENSG00000187243",  # MAGED4B cDNA-identical group member
                "ENSG00000141510",
            ],
            "Symbol": ["MAGED4", "MAGED4B", "TP53"],
            "p25": [1.0, 10.0, 100.0],
            "p50": [2.0, 20.0, 200.0],
            "p75": [3.0, 30.0, 300.0],
        }
    )
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    _mock_missing_artifact_build_metadata(monkeypatch)
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(expression, "cohort_gene_percentiles", lambda *a, **k: pct.copy())

    out = expression.cancer_reference_expression("x", include_provenance=False)

    assert list(out.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "Proteoform_ID",
        "Member_Ensembl_Gene_IDs",
        "cancer_code",
        "normalization",
        "expression",
        "q1",
        "q3",
    ]
    assert len(out) == 3
    by_id = out.set_index("Ensembl_Gene_ID")
    assert by_id.loc["ENSG00000154545", "Proteoform_ID"] == "MAGED4"
    assert by_id.loc["ENSG00000187243", "Proteoform_ID"] == "MAGED4"
    assert by_id.loc["ENSG00000154545", "Member_Ensembl_Gene_IDs"] == "ENSG00000154545"
    assert by_id.loc["ENSG00000187243", "Member_Ensembl_Gene_IDs"] == "ENSG00000187243"
    assert by_id.loc["ENSG00000141510", "Proteoform_ID"] == "ENSG00000141510"


def test_cancer_reference_expression_cdna_identical_collapse(monkeypatch):
    pct = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [
                "ENSG00000154545",  # MAGED4 cDNA-identical group member
                "ENSG00000187243",  # MAGED4B cDNA-identical group member
                "ENSG00000141510",
            ],
            "Symbol": ["MAGED4", "MAGED4B", "TP53"],
            "p25": [1.0, 10.0, 100.0],
            "p50": [2.0, 20.0, 200.0],
            "p75": [3.0, 30.0, 300.0],
        }
    )
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    _mock_missing_artifact_build_metadata(monkeypatch)
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(expression, "cohort_gene_percentiles", lambda *a, **k: pct.copy())

    out = expression.cancer_reference_expression(
        "x", include_provenance=False, collapse_cdna_identical=True
    )

    assert list(out.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "Proteoform_ID",
        "Member_Ensembl_Gene_IDs",
        "cancer_code",
        "normalization",
        "expression",
        "q1",
        "q3",
    ]
    maged4 = out.set_index("Ensembl_Gene_ID").loc["MAGED4"]
    assert maged4["Symbol"] == "MAGED4"
    assert maged4["Proteoform_ID"] == "MAGED4"
    assert maged4["Member_Ensembl_Gene_IDs"] == "ENSG00000154545;ENSG00000187243"
    assert maged4["q1"] == 11.0
    assert maged4["expression"] == 22.0
    assert maged4["q3"] == 33.0
    tp53 = out.set_index("Ensembl_Gene_ID").loc["ENSG00000141510"]
    assert tp53["Proteoform_ID"] == "ENSG00000141510"
    assert tp53["Member_Ensembl_Gene_IDs"] == "ENSG00000141510"


def test_reference_summary_collapse_aggregates_n_detected_outside_identity_key():
    ref = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000154545", "ENSG00000187243"],
            "Symbol": ["MAGED4", "MAGED4B"],
            "source_cohort": ["SRC", "SRC"],
            "n_reference_samples": [10, 10],
            "n_samples": [10, 10],
            "n_detected": [2, 7],
            "p25": [1.0, 10.0],
            "p50": [2.0, 20.0],
            "p75": [3.0, 30.0],
        }
    )

    group_keys = expression._reference_collapse_group_keys(ref)
    assert "n_detected" not in group_keys
    out = expression._collapse_reference_identical_loci(
        ref,
        kind="cdna",
        group_keys=group_keys,
    )

    assert len(out) == 1
    assert out.loc[0, "Ensembl_Gene_ID"] == "MAGED4"
    assert out.loc[0, "p50"] == pytest.approx(22.0)
    assert out.loc[0, "n_detected"] == 7


def test_reference_summary_collapse_uses_only_observed_categorical_groups():
    ref = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000154545", "ENSG00000187243"],
            "Symbol": ["MAGED4", "MAGED4B"],
            "cancer_code": pd.Categorical(["X", "X"], categories=["X", "UNUSED_CODE"]),
            "source_cohort": pd.Categorical(["SRC", "SRC"], categories=["SRC", "UNUSED_SOURCE"]),
            "n_detected": [2, 7],
            "p25": [1.0, 10.0],
            "p50": [2.0, 20.0],
            "p75": [3.0, 30.0],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = expression._collapse_reference_identical_loci(
            ref,
            kind="cdna",
            group_keys=["cancer_code", "source_cohort"],
        )

    assert len(out) == 1
    assert out.loc[0, "cancer_code"] == "X"
    assert out.loc[0, "source_cohort"] == "SRC"
    assert out.loc[0, "p50"] == pytest.approx(22.0)
    assert out.loc[0, "n_detected"] == 7


def test_pirlygenes_identity_style_preserves_legacy_compact_group_ids():
    cdna, _ = expression._identical_locus_identity_maps("cdna", "pirlygenes")
    protein, _ = expression._identical_locus_identity_maps("protein", "pirlygenes")

    assert cdna["ENSG00000184033"] == "CTAG1A/B"
    assert cdna["ENSG00000187243"] == "MAGED4/MAGED4B"
    assert protein["ENSG00000184033"] == "CTAG1A/B"
    assert protein["ENSG00000183889"] == "NPIPA6/9"
    assert "ENSG00000169789" not in protein  # PRY group post-dates the legacy snapshot.


def test_cancer_reference_expression_uses_pirlygenes_identity_style(monkeypatch):
    pct = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000268651", "ENSG00000184033"],
            "Symbol": ["CTAG1A", "CTAG1B"],
            "p25": [1.0, 4.0],
            "p50": [2.0, 5.0],
            "p75": [3.0, 6.0],
        }
    )
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    _mock_missing_artifact_build_metadata(monkeypatch)
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(expression, "cohort_gene_percentiles", lambda *a, **k: pct.copy())

    bridge = expression.cancer_reference_expression(
        "x", include_provenance=False, gene_id_style="pirlygenes"
    )
    assert set(bridge["Proteoform_ID"]) == {"CTAG1A/B"}

    collapsed = expression.cancer_reference_expression(
        "x",
        include_provenance=False,
        gene_id_style="pirlygenes",
        collapse_cdna_identical=True,
    )
    assert collapsed.loc[0, "Ensembl_Gene_ID"] == "CTAG1A/B"
    assert collapsed.loc[0, "expression"] == pytest.approx(7.0)


def test_cancer_reference_expression_protein_identical_collapse_and_wide_shape(monkeypatch):
    pct = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [
                "ENSG00000268651",  # CTAG1A protein-identical group member
                "ENSG00000184033",  # CTAG1B protein-identical group member
            ],
            "Symbol": ["CTAG1A", "CTAG1B"],
            "p25": [1.0, 4.0],
            "p50": [2.0, 5.0],
            "p75": [3.0, 6.0],
        }
    )
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    _mock_missing_artifact_build_metadata(monkeypatch)
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(expression, "cohort_gene_percentiles", lambda *a, **k: pct.copy())

    long = expression.cancer_reference_expression(
        "x", include_provenance=False, collapse_protein_identical=True
    )
    row = long.set_index("Ensembl_Gene_ID").loc["CTAG1A/CTAG1B"]
    assert row["Symbol"] == "CTAG1A/CTAG1B"
    assert row["Proteoform_ID"] == "CTAG1A/CTAG1B"
    assert row["Member_Ensembl_Gene_IDs"] == "ENSG00000184033;ENSG00000268651"
    assert row["expression"] == 7.0

    wide = expression.cancer_reference_expression(
        "x", format="wide", collapse_protein_identical=True
    )
    assert list(wide.columns) == ["Ensembl_Gene_ID", "Symbol", "X_TPM_clean"]
    assert wide.loc[0, "Ensembl_Gene_ID"] == "CTAG1A/CTAG1B"
    assert wide.loc[0, "X_TPM_clean"] == 7.0


def test_cancer_reference_expression_rejects_two_collapse_modes():
    with pytest.raises(ValueError, match="at most one"):
        expression.cancer_reference_expression(
            "PRAD",
            collapse_cdna_identical=True,
            collapse_protein_identical=True,
        )


def test_cancer_reference_expression_multiple_normalizations(monkeypatch):
    pct_tpm = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "p25": [1.0],
            "p50": [3.0],
            "p75": [5.0],
        }
    )
    pct_log = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "p25": [np.log1p(1.0)],
            "p50": [np.log1p(3.0)],
            "p75": [np.log1p(5.0)],
        }
    )

    def fake_pct(code, *, as_tpm=True, **kwargs):
        return pct_tpm.copy() if as_tpm else pct_log.copy()

    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    _mock_missing_artifact_build_metadata(monkeypatch)
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(expression, "cohort_gene_percentiles", fake_pct)

    long = expression.cancer_reference_expression("x", normalize=["clean_tpm", "tpm_clean_log1p"])
    assert long["normalization"].tolist() == ["tpm_clean", "tpm_clean_log1p"]
    assert long["expression"].tolist() == [3.0, pytest.approx(np.log1p(3.0))]

    wide = expression.cancer_reference_expression(
        "x", normalize=["tpm_clean", "tpm_clean_log1p"], format="wide"
    )
    assert wide.loc[0, "X_TPM_clean"] == 3.0
    assert wide.loc[0, "X_TPM_clean_log1p"] == pytest.approx(np.log1p(3.0))


def test_cancer_reference_expression_availability_reports_missing(monkeypatch):
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression.source_matrices, "available_cohorts", lambda: ["Y"])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    _mock_missing_artifact_build_metadata(monkeypatch)

    out = expression.cancer_reference_expression_availability(
        ["x", "z"], normalize=["tpm_clean", "tpm_raw"]
    )

    assert list(out.columns) == [
        "requested_code",
        "cancer_code",
        "request_kind",
        "normalization",
        "available",
        "missing_reason",
        "source_cohort",
        "source_project",
        "source_version",
        "source_type",
        "source_unit",
        "source_scale_class",
        "linear_tpm_comparable",
        "tumor_origin",
        "metastasis_site",
        "n_reference_genes",
        "n_reference_samples",
        "n_samples",
        "n_detected",
        "processing_pipeline",
        "notes",
        "reference_method",
        "sample_qc",
        "sample_qc_effective",
        "artifact_sample_qc",
        "artifact_schema_version",
        "data_version",
        "source_matrix_version",
    ]
    assert out.attrs["artifact_schema_version"] == expression.REFERENCE_EXPRESSION_SCHEMA_VERSION
    keyed = out.set_index(["cancer_code", "normalization"])
    assert bool(keyed.loc[("X", "tpm_clean"), "available"]) is True
    assert keyed.loc[("X", "tpm_clean"), "sample_qc"] == "pass"
    assert pd.isna(keyed.loc[("X", "tpm_clean"), "artifact_sample_qc"])
    assert keyed.loc[("X", "tpm_raw"), "sample_qc"] == "pass"
    assert keyed.loc[("X", "tpm_raw"), "missing_reason"] == "no_source_matrix"
    assert keyed.loc[("Z", "tpm_clean"), "missing_reason"] == "no_percentile_artifact"


def test_cancer_reference_expression_availability_matches_artifact_qc_policy(
    percentile_cache,
):
    shard_path = percentile_cache / "cancer-reference-expression-percentiles" / "PRAD.parquet"
    shard = pd.read_parquet(shard_path)
    shard["p25"] = np.log1p([25.0, 50.0]).astype("float16")
    shard["p75"] = np.log1p([75.0, 150.0]).astype("float16")
    shard.to_parquet(shard_path, index=False)
    _write_artifact_build_metadata(percentile_cache, "PRAD", sample_qc="pass_or_warn")

    strict = expression.cancer_reference_expression_availability("PRAD", sample_qc="pass")
    assert strict["available"].tolist() == [False]
    assert strict["missing_reason"].tolist() == ["artifact_sample_qc_mismatch"]
    assert strict["sample_qc"].tolist() == ["pass"]
    assert strict["artifact_sample_qc"].tolist() == ["pass_or_warn"]

    strict_load = expression.cancer_reference_expression(
        "PRAD", sample_qc="pass", on_missing="empty"
    )
    assert strict_load.empty
    assert strict_load.attrs["missing_requests"][0]["missing_reason"] == (
        "artifact_sample_qc_mismatch"
    )

    compatible = expression.cancer_reference_expression_availability(
        "PRAD", sample_qc="pass_or_warn"
    )
    assert compatible["available"].tolist() == [True]
    assert compatible["artifact_sample_qc"].tolist() == ["pass_or_warn"]
    assert not expression.cancer_reference_expression(
        "PRAD", sample_qc="pass_or_warn", on_missing="empty"
    ).empty

    audit = expression.cancer_reference_expression_availability("PRAD", sample_qc="artifact")
    assert audit["available"].tolist() == [True]
    assert audit["artifact_sample_qc"].tolist() == ["pass_or_warn"]


def test_cancer_reference_expression_availability_rejects_missing_artifact_metadata_row(
    monkeypatch,
):
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(
        expression,
        "expression_artifact_build_metadata",
        lambda *args, **kwargs: pd.DataFrame(
            {"cancer_code": ["Y"], "sample_qc_effective": ["pass"]}
        ),
    )

    out = expression.cancer_reference_expression_availability("X", sample_qc="pass")

    assert out["available"].tolist() == [False]
    assert out["missing_reason"].tolist() == ["artifact_build_metadata_missing"]
    assert out["artifact_sample_qc"].isna().all()


def test_cancer_reference_expression_missing_empty_and_raise(monkeypatch):
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression.source_matrices, "available_cohorts", lambda: [])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())

    empty = expression.cancer_reference_expression("z", on_missing="empty")
    assert empty.empty
    assert list(empty.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "Proteoform_ID",
        "Member_Ensembl_Gene_IDs",
        "cancer_code",
        "normalization",
        "source_cohort",
        "source_project",
        "source_version",
        "source_type",
        "source_unit",
        "source_scale_class",
        "linear_tpm_comparable",
        "tumor_origin",
        "metastasis_site",
        "n_reference_genes",
        "n_reference_samples",
        "n_samples",
        "n_detected",
        "processing_pipeline",
        "notes",
        "reference_method",
        "sample_qc",
        "sample_qc_effective",
        "data_version",
        "source_matrix_version",
        "expression",
        "q1",
        "q3",
    ]
    assert empty.attrs["missing_requests"][0]["cancer_code"] == "Z"
    assert empty.attrs["missing_requests"][0]["missing_reason"] == "no_percentile_artifact"
    with pytest.raises(ValueError, match="Z/tpm_clean: no_percentile_artifact"):
        expression.cancer_reference_expression("z", on_missing="raise")


def test_cancer_reference_expression_request_metadata_for_aggregate(monkeypatch):
    pct = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "p25": [1.0],
            "p50": [3.0],
            "p75": [5.0],
        }
    )
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X", "Y"])
    _mock_missing_artifact_build_metadata(monkeypatch)
    monkeypatch.setattr(expression, "cohort_aggregates", lambda: {"AGG": ["X", "Y"]})
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(expression, "cohort_gene_percentiles", lambda *a, **k: pct.copy())

    out = expression.cancer_reference_expression("agg", include_request_metadata=True)

    assert out["requested_code"].tolist() == ["AGG", "AGG"]
    assert out["cancer_code"].tolist() == ["X", "Y"]
    assert out["request_kind"].tolist() == ["aggregate_member", "aggregate_member"]
    assert out["available"].tolist() == [True, True]
    assert out["missing_reason"].tolist() == ["", ""]


def test_cancer_reference_expression_raw_tpm_uses_source_stats(monkeypatch):
    stats = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "p25": [10.0],
            "p50": [20.0],
            "p75": [30.0],
        }
    )
    seen = {}
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression.source_matrices, "available_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(
        expression,
        "cohort_stats",
        lambda code, **k: seen.update(k) or stats.copy(),
    )
    monkeypatch.setattr(
        expression,
        "_selected_expression_source_metadata",
        lambda code: {
            "source_cohort": "SRC_X",
            "source_type": "gdc",
            "unit": "TPM",
            "source_scale_class": "linear_rnaseq_tpm",
            "linear_tpm_comparable": True,
            "tpm_proxy": False,
        },
    )

    long = expression.cancer_reference_expression("x", normalize="tpm", auto_fetch=True)
    assert seen == {"normalize": "tpm_raw", "auto_fetch": True, "sample_qc": "pass"}
    assert long["normalization"].tolist() == ["tpm_raw"]
    assert long["sample_qc"].tolist() == ["pass"]

    expression.cancer_reference_expression("x", normalize="tpm", auto_fetch=True, sample_qc="all")
    assert seen == {"normalize": "tpm_raw", "auto_fetch": True, "sample_qc": "all"}
    assert long["source_cohort"].tolist() == ["SRC_X"]
    assert long["source_type"].tolist() == ["gdc"]
    assert long["reference_method"].tolist() == ["source_matrix_stats"]
    assert long["expression"].tolist() == [20.0]


def test_cancer_reference_expression_accepts_artifact_qc_for_clean_shards(monkeypatch):
    pct = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "p25": [1.0],
            "p50": [2.0],
            "p75": [3.0],
        }
    )
    seen = {}
    metadata = pd.DataFrame({"cancer_code": ["X"], "sample_qc_effective": ["pass_or_warn"]})
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(
        expression,
        "expression_artifact_build_metadata",
        lambda *args, **kwargs: metadata,
    )
    monkeypatch.setattr(
        expression,
        "cohort_gene_percentiles",
        lambda code, **kwargs: seen.update(kwargs) or pct.copy(),
    )

    out = expression.cancer_reference_expression("x", sample_qc="artifact")

    assert seen["sample_qc"] == "artifact"
    assert out["sample_qc"].tolist() == ["artifact"]
    assert out["sample_qc_effective"].tolist() == ["pass_or_warn"]


def test_cancer_reference_expression_uses_per_code_artifact_qc_for_raw_tpm(monkeypatch):
    stats = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "p25": [10.0],
            "p50": [20.0],
            "p75": [30.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "cancer_code": ["X", "Y"],
            "sample_qc_effective": ["pass", "pass_or_warn"],
            "n_cohort_samples": [3, 4],
        }
    )
    calls = []
    monkeypatch.setattr(expression.source_matrices, "available_cohorts", lambda: ["X", "Y"])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(expression, "cohort_aggregates", lambda: {})
    monkeypatch.setattr(
        expression,
        "expression_artifact_build_metadata",
        lambda *args, **kwargs: metadata,
    )
    monkeypatch.setattr(
        expression,
        "_source_matrix_effective_sample_count",
        lambda code, sample_qc: None,
    )
    monkeypatch.setattr(
        expression,
        "cohort_stats",
        lambda code, **kwargs: calls.append((code, kwargs["sample_qc"])) or stats.copy(),
    )
    monkeypatch.setattr(
        expression,
        "_selected_expression_source_metadata",
        lambda code: {
            "source_cohort": f"SRC_{code}",
            "source_type": "gdc",
            "unit": "TPM",
            "source_scale_class": "linear_rnaseq_tpm",
            "linear_tpm_comparable": True,
        },
    )

    availability = expression.cancer_reference_expression_availability(
        ["x", "y"], normalize="tpm_raw", sample_qc="artifact"
    ).set_index("cancer_code")
    assert availability["artifact_sample_qc"].to_dict() == {
        "X": "pass",
        "Y": "pass_or_warn",
    }
    assert availability["sample_qc_effective"].to_dict() == {
        "X": "pass",
        "Y": "pass_or_warn",
    }
    assert availability["n_reference_samples"].to_dict() == {"X": 3, "Y": 4}

    out = expression.cancer_reference_expression(
        ["x", "y"], normalize="tpm_raw", sample_qc="artifact"
    )

    assert calls == [("X", "pass"), ("Y", "pass_or_warn")]
    provenance = out.set_index("cancer_code")
    assert provenance["sample_qc"].to_dict() == {"X": "artifact", "Y": "artifact"}
    assert provenance["sample_qc_effective"].to_dict() == {
        "X": "pass",
        "Y": "pass_or_warn",
    }


def test_raw_artifact_qc_reports_missing_build_metadata(monkeypatch):
    metadata = pd.DataFrame()
    metadata.attrs["missing_reason"] = "expression artifact build metadata not present in bundle"
    monkeypatch.setattr(expression.source_matrices, "available_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(
        expression,
        "expression_artifact_build_metadata",
        lambda *args, **kwargs: metadata,
    )

    availability = expression.cancer_reference_expression_availability(
        "x", normalize="tpm_raw", sample_qc="artifact"
    )

    assert availability["available"].tolist() == [False]
    assert availability["missing_reason"].tolist() == ["artifact_build_metadata_missing"]
    assert pd.isna(availability.loc[0, "sample_qc_effective"])


def test_cancer_reference_expression_rejects_artifact_qc_for_summary_rows():
    with pytest.raises(ValueError, match="requires reference_source='artifact'"):
        expression.cancer_reference_expression(
            "PRAD", reference_source="summary_rows", sample_qc="artifact"
        )


def test_cancer_reference_expression_summary_rows_selects_richest_source(monkeypatch):
    expression._reference_summary_source_table.cache_clear()
    summary = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E2", "E1", "E2", "E3"],
            "Symbol": ["A", "B", "A", "B", "C"],
            "cancer_code": ["X", "X", "X", "X", "X"],
            "source_cohort": ["SMALL", "SMALL", "RICH", "RICH", "RICH"],
            "source_project": ["GEO_SMALL", "GEO_SMALL", "GEO_RICH", "GEO_RICH", "GEO_RICH"],
            "source_version": ["v1", "v1", "v2", "v2", "v2"],
            "TPM_median": [100.0, 200.0, 10.0, 20.0, 30.0],
            "TPM_q1": [90.0, 190.0, 9.0, 19.0, 29.0],
            "TPM_q3": [110.0, 210.0, 11.0, 21.0, 31.0],
            "TPM_clean_median": [50.0, 60.0, 1.0, 2.0, 3.0],
            "TPM_clean_q1": [45.0, 55.0, 0.5, 1.5, 2.5],
            "TPM_clean_q3": [55.0, 65.0, 1.5, 2.5, 3.5],
            "n_samples": [20, 20, 5, 5, 5],
            "tumor_origin": ["primary", "primary", "primary", "primary", "primary"],
            "metastasis_site": [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
        }
    )
    summary_availability = pd.DataFrame(
        {
            "cancer_code": ["X", "X"],
            "source_cohort": ["SMALL", "RICH"],
            "source_project": ["GEO_SMALL", "GEO_RICH"],
            "source_version": ["v1", "v2"],
            "n_reference_genes": [2, 3],
            "n_reference_samples": [20, 5],
            "selected": [False, True],
        }
    )
    _mock_reference_summary_shards(monkeypatch, summary)
    monkeypatch.setattr(
        expression,
        "_reference_summary_availability_table",
        lambda: summary_availability,
    )
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: [])
    monkeypatch.setattr(expression.source_matrices, "available_cohorts", lambda: [])
    monkeypatch.setattr(
        expression,
        "cohort_stats",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should use summary rows")),
    )

    out = expression.cancer_reference_expression(
        "x", reference_source="summary_rows", sample_qc="all"
    )

    assert out.attrs["reference_source"] == "summary_rows"
    assert out["source_cohort"].unique().tolist() == ["RICH"]
    assert out["source_project"].unique().tolist() == ["GEO_RICH"]
    assert out["source_version"].unique().tolist() == ["v2"]
    assert out["reference_method"].unique().tolist() == ["source_summary_rows"]
    assert out["n_reference_genes"].unique().tolist() == [3]
    assert out["n_reference_samples"].unique().tolist() == [5]
    keyed = out.set_index("Ensembl_Gene_ID")
    assert keyed.loc["E1", "expression"] == pytest.approx(1.0)
    assert keyed.loc["E1", "q1"] == pytest.approx(0.5)
    assert keyed.loc["E1", "q3"] == pytest.approx(1.5)

    raw_log = expression.cancer_reference_expression(
        "x",
        genes="E1",
        normalize=["tpm_raw", "tpm_clean_log1p"],
        reference_source="summary_rows",
        sample_qc="all",
    ).set_index("normalization")
    assert raw_log.loc["tpm_raw", "expression"] == pytest.approx(10.0)
    assert raw_log.loc["tpm_clean_log1p", "expression"] == pytest.approx(np.log1p(1.0))
    expression._reference_summary_source_table.cache_clear()


def test_cancer_reference_expression_reads_only_bounded_summary_shards(monkeypatch, tmp_path):
    shard_dir = tmp_path / "cancer-reference-expression"
    shard_dir.mkdir()
    columns = {
        "Symbol": ["A", "B", "A"],
        "cancer_code": ["X", "X", "Y"],
        "source_project": ["TEST"] * 3,
        "source_version": ["v1"] * 3,
        "TPM_median": [10.0, 20.0, 999.0],
        "TPM_q1": [9.0, 19.0, 998.0],
        "TPM_q3": [11.0, 21.0, 1000.0],
        "TPM_clean_median": [1.0, 2.0, 99.0],
        "TPM_clean_q1": [0.9, 1.9, 98.0],
        "TPM_clean_q3": [1.1, 2.1, 100.0],
        "n_samples": [4, 4, 1],
        "n_detected": [4, 3, 1],
        "processing_pipeline": ["rna_seq"] * 3,
        "notes": [""] * 3,
        "tumor_origin": ["primary"] * 3,
        "metastasis_site": [pd.NA] * 3,
    }
    for source, filename in (("SRC_A", "SRC_A.csv.gz"), ("SRC_B", "SRC_B__X.csv.gz")):
        frame = pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["E1", "E2", "E1"],
                "source_cohort": [source] * 3,
                **columns,
            }
        )
        frame.to_csv(shard_dir / filename, index=False, compression="gzip")

    availability = pd.DataFrame(
        {
            "cancer_code": ["X", "X"],
            "source_cohort": ["SRC_A", "SRC_B"],
            "source_project": ["TEST", "TEST"],
            "source_version": ["v1", "v1"],
            "tumor_origin": ["primary", "primary"],
            "n_reference_genes": [2, 2],
            "n_reference_samples": [4, 4],
            "processing_pipeline": ["rna_seq", "rna_seq"],
            "selected": [True, False],
        }
    )
    expression._reference_summary_source_table.cache_clear()
    monkeypatch.setattr(expression, "_reference_summary_availability_table", lambda: availability)
    monkeypatch.setattr(expression, "_bundle_subdir", lambda *args, **kwargs: shard_dir)
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code, **kwargs: str(code).upper())
    read_paths = []
    read_csv = pd.read_csv

    def recording_read_csv(path, *args, **kwargs):
        path = Path(path)
        if path.parent == shard_dir:
            read_paths.append(path)
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(expression.pd, "read_csv", recording_read_csv)
    out = expression.cancer_reference_expression(
        "x",
        genes="E1",
        normalize=["tpm_raw", "tpm_clean"],
        reference_source="summary_rows_all",
        sample_qc="all",
    )

    assert set(out["Ensembl_Gene_ID"]) == {"E1"}
    assert set(out["cancer_code"]) == {"X"}
    assert set(out["source_cohort"]) == {"SRC_A", "SRC_B"}
    assert set(out["normalization"]) == {"tpm_raw", "tpm_clean"}
    assert set(out["n_reference_genes"]) == {2}
    assert set(out["n_reference_samples"]) == {4}
    assert sorted(path.name for path in read_paths) == [
        "SRC_A.csv.gz",
        "SRC_B__X.csv.gz",
    ]
    expression._reference_summary_source_table.cache_clear()


@pytest.mark.parametrize(
    ("code", "source_cohort", "legacy_filename"),
    [
        (
            "SARC_PLEOLPS",
            "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY",
            "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES__SARC_PLEOLPS.csv.gz",
        ),
        (
            "SARC_DDLPS",
            "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY",
            "TREEHOUSE_POLYA_25_01_TCGA_SUBSET__SARC_DDLPS.csv.gz",
        ),
        (
            "LUAD",
            "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES",
            "TREEHOUSE_POLYA_25_01_TCGA_SUBSET__LUAD.csv.gz",
        ),
    ],
)
def test_reference_summary_shard_path_supports_legacy_treehouse_names(
    tmp_path, code, source_cohort, legacy_filename
):
    legacy_path = tmp_path / legacy_filename
    legacy_path.touch()

    assert expression._reference_summary_shard_path(tmp_path, code, source_cohort) == legacy_path


def test_cancer_reference_expression_summary_rows_all_preserves_sources_and_filters(monkeypatch):
    expression._reference_summary_source_table.cache_clear()
    expression._source_cohort_kind_map.cache_clear()
    expression._source_cohort_metadata_map.cache_clear()
    summary = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E1", "E2", "E1"],
            "Symbol": ["A", "A", "B", "A"],
            "cancer_code": ["X", "X", "X", "Y"],
            "source_cohort": ["GEO_X", "TREE_X", "TREE_X", "GEO_Y"],
            "source_project": ["GEO", "Treehouse", "Treehouse", "GEO"],
            "source_version": ["v1", "v2", "v2", "v1"],
            "TPM_median": [10.0, 20.0, 30.0, 40.0],
            "TPM_q1": [9.0, 19.0, 29.0, 39.0],
            "TPM_q3": [11.0, 21.0, 31.0, 41.0],
            "TPM_clean_median": [1.0, 2.0, 3.0, 4.0],
            "TPM_clean_q1": [0.9, 1.9, 2.9, 3.9],
            "TPM_clean_q3": [1.1, 2.1, 3.1, 4.1],
            "n_samples": [4, 8, 8, 3],
            "n_detected": [3, 8, 7, 2],
            "processing_pipeline": [
                "geo_microarray_tpm_proxy_clean_tpm_16_9_75",
                "treehouse_polya_tpm_clean_tpm_16_9_75",
                "treehouse_polya_tpm_clean_tpm_16_9_75",
                "geo_microarray_tpm_proxy_clean_tpm_16_9_75",
            ],
            "notes": ["geo notes", "tree notes", "tree notes", "geo y notes"],
            "tumor_origin": ["primary", "mixed", "mixed", "primary"],
            "metastasis_site": [pd.NA, pd.NA, pd.NA, pd.NA],
        }
    )
    summary_availability = pd.DataFrame(
        {
            "cancer_code": ["X", "X", "Y"],
            "source_cohort": ["GEO_X", "TREE_X", "GEO_Y"],
            "source_project": ["GEO", "Treehouse", "GEO"],
            "source_version": ["v1", "v2", "v1"],
            "tumor_origin": ["primary", "mixed", "primary"],
            "metastasis_site": [pd.NA, pd.NA, pd.NA],
            "n_reference_genes": [1, 2, 1],
            "n_reference_samples": [4, 8, 3],
            "processing_pipeline": [
                "geo_microarray_tpm_proxy_clean_tpm_16_9_75",
                "treehouse_polya_tpm_clean_tpm_16_9_75",
                "geo_microarray_tpm_proxy_clean_tpm_16_9_75",
            ],
            "notes": ["geo notes", "tree notes", "geo y notes"],
            "selected": [False, True, True],
        }
    )
    registry = pd.DataFrame(
        {
            "cohort_id": ["GEO_X", "TREE_X", "GEO_Y"],
            "kind": ["geo", "treehouse", "geo"],
        }
    )
    _mock_reference_summary_shards(monkeypatch, summary)
    monkeypatch.setattr(
        expression,
        "_reference_summary_availability_table",
        lambda: summary_availability,
    )
    monkeypatch.setattr(expression, "cohort_registry_df", lambda: registry)
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code, **k: str(code).upper())
    monkeypatch.setattr(expression.source_matrices, "available_cohorts", lambda: [])

    out = expression.cancer_reference_expression(
        "x",
        genes="E1",
        reference_source="summary_rows_all",
        sample_qc="all",
    )

    assert out.attrs["reference_source"] == "summary_rows_all"
    assert out["source_cohort"].tolist() == ["GEO_X", "TREE_X"]
    assert out["processing_pipeline"].tolist() == [
        "geo_microarray_tpm_proxy_clean_tpm_16_9_75",
        "treehouse_polya_tpm_clean_tpm_16_9_75",
    ]
    assert out["n_samples"].tolist() == [4, 8]
    assert out["n_detected"].tolist() == [3, 8]
    assert out["reference_method"].unique().tolist() == ["source_summary_rows_all"]
    assert out["sample_qc_effective"].unique().tolist() == ["all"]

    compact = expression.cancer_reference_expression(
        "x",
        genes="E1",
        reference_source="summary_rows_all",
        sample_qc="all",
        include_provenance=False,
    )
    assert compact["source_cohort"].tolist() == ["GEO_X", "TREE_X"]
    assert compact["n_reference_samples"].tolist() == [4, 8]
    assert compact["n_samples"].tolist() == [4, 8]
    assert "processing_pipeline" not in compact.columns

    with pytest.raises(ValueError, match='requires format="long"'):
        expression.cancer_reference_expression(
            "x",
            reference_source="summary_rows_all",
            sample_qc="all",
            format="wide",
        )

    tree = expression.cancer_reference_expression(
        "x",
        genes="E1",
        reference_source="summary_rows_all",
        sample_qc="all",
        source_kind="treehouse",
    )
    assert tree["source_cohort"].tolist() == ["TREE_X"]

    non_proxy = expression.cancer_reference_expression(
        "x",
        genes="E1",
        reference_source="summary_rows_all",
        sample_qc="all",
        exclude_microarray_proxy=True,
    )
    assert non_proxy["source_cohort"].tolist() == ["TREE_X"]

    all_tree = expression.cancer_reference_expression(
        reference_source="summary_rows_all",
        sample_qc="all",
        source_kind="treehouse",
        on_missing="raise",
    )
    assert all_tree["cancer_code"].unique().tolist() == ["X"]
    assert all_tree["source_cohort"].unique().tolist() == ["TREE_X"]

    filtered_availability = expression.cancer_reference_expression_availability(
        reference_source="summary_rows_all",
        sample_qc="all",
        source_kind="treehouse",
    )
    assert filtered_availability["cancer_code"].tolist() == ["X"]
    assert filtered_availability["available"].tolist() == [True]

    empty = expression.cancer_reference_expression_availability(
        "x",
        reference_source="summary_rows_all",
        sample_qc="all",
        source_kind="curated",
    )
    assert bool(empty.loc[0, "available"]) is False
    assert empty.loc[0, "missing_reason"] == "no_reference_summary_rows"

    selected = expression.cancer_reference_expression(
        ["x", "y"],
        genes="E1",
        reference_source="summary_rows_all",
        sample_qc="all",
        source_kind="treehouse",
        on_missing="empty",
    )
    availability = pd.DataFrame(selected.attrs["availability"]).set_index("cancer_code")
    assert availability.loc["X", "source_cohort"] == "TREE_X"
    assert bool(availability.loc["X", "available"]) is True
    assert pd.isna(availability.loc["Y", "source_cohort"])
    assert bool(availability.loc["Y", "available"]) is False
    assert availability.loc["Y", "missing_reason"] == "no_reference_summary_rows"

    expression._reference_summary_source_table.cache_clear()
    expression._source_cohort_kind_map.cache_clear()
    expression._source_cohort_metadata_map.cache_clear()


def test_source_filter_normalization_distinguishes_unfiltered_from_no_matches():
    assert expression._normalize_source_filter_values(None) is None
    assert expression._normalize_source_filter_values([]) == set()
    assert expression._normalize_source_filter_values("") == set()
    assert expression._normalize_source_cohort_filter_values(None) is None
    assert expression._normalize_source_cohort_filter_values([]) == set()


@pytest.mark.parametrize(
    "filter_kwargs",
    [
        pytest.param({"source_cohort": []}, id="empty-cohort-list"),
        pytest.param({"source_cohort": ""}, id="blank-cohort-scalar"),
        pytest.param({"source_kind": []}, id="empty-kind-list"),
    ],
)
@pytest.mark.parametrize("pool", [False, True])
def test_explicit_empty_source_filters_match_nothing(monkeypatch, filter_kwargs, pool):
    summary = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "cancer_code": ["X"],
            "source_cohort": ["SRC_X"],
            "source_project": ["TEST"],
            "source_version": ["v1"],
            "TPM_median": [10.0],
            "TPM_q1": [9.0],
            "TPM_q3": [11.0],
            "TPM_clean_median": [1.0],
            "TPM_clean_q1": [0.9],
            "TPM_clean_q3": [1.1],
            "n_samples": [4],
            "n_detected": [3],
            "processing_pipeline": ["rna_seq"],
            "notes": [""],
            "tumor_origin": ["primary"],
            "metastasis_site": [pd.NA],
        }
    )
    compact = pd.DataFrame(
        {
            "cancer_code": ["X"],
            "source_cohort": ["SRC_X"],
            "n_reference_genes": [1],
            "n_reference_samples": [4],
        }
    )
    registry = pd.DataFrame({"cohort_id": ["SRC_X"], "kind": ["test"]})
    expression._reference_summary_source_table.cache_clear()
    expression._source_cohort_kind_map.cache_clear()
    expression._source_cohort_metadata_map.cache_clear()
    _mock_reference_summary_shards(monkeypatch, summary)
    monkeypatch.setattr(expression, "_reference_summary_availability_table", lambda: compact)
    monkeypatch.setattr(expression, "cohort_registry_df", lambda: registry)
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code, **k: str(code).upper())
    monkeypatch.setattr(expression.source_matrices, "available_cohorts", lambda: [])

    out = expression.cancer_reference_expression(
        "x",
        reference_source="summary_rows_all",
        sample_qc="all",
        on_missing="empty",
        pool=pool,
        **filter_kwargs,
    )

    assert out.empty
    assert len(out.attrs["availability"]) == 1
    assert out.attrs["availability"][0]["available"] is False
    assert out.attrs["availability"][0]["missing_reason"] == "no_reference_summary_rows"
    assert len(out.attrs["missing_requests"]) == 1
    assert out.attrs["missing_requests"][0]["cancer_code"] == "X"
    assert out.attrs["missing_requests"][0]["missing_reason"] == "no_reference_summary_rows"

    availability = expression.cancer_reference_expression_availability(
        "x",
        reference_source="summary_rows_all",
        sample_qc="all",
        all_sources=True,
        **filter_kwargs,
    )
    assert len(availability) == 1
    assert not bool(availability.loc[0, "available"])
    assert availability.loc[0, "missing_reason"] == "no_reference_summary_rows"

    expression._reference_summary_source_table.cache_clear()
    expression._source_cohort_kind_map.cache_clear()
    expression._source_cohort_metadata_map.cache_clear()


def test_summary_provenance_uses_only_observed_source_categories(monkeypatch):
    summary = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E2"],
            "source_cohort": pd.Categorical(["SRC", "SRC"], categories=["SRC", "UNUSED_SOURCE"]),
            "n_samples": [4, 4],
        }
    )
    monkeypatch.setattr(
        expression,
        "_selected_expression_source_metadata",
        lambda code, *, source_cohort=None: {
            "source_type": "geo",
            "unit": "TPM",
            "source_scale_class": "linear_rnaseq_tpm",
            "linear_tpm_comparable": True,
        },
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = expression._attach_summary_row_provenance(summary, code="X")

    assert out["source_cohort"].astype("string").tolist() == ["SRC", "SRC"]
    assert out["n_reference_genes"].tolist() == [2, 2]
    assert out["n_reference_samples"].tolist() == [4, 4]


def test_reference_availability_all_sources_uses_compact_manifest(monkeypatch):
    sidecar = pd.DataFrame(
        {
            "cancer_code": [
                "SARC_DDLPS",
                "SARC_DDLPS",
                "SARC_DDLPS",
                "NET_PANCREAS",
                "NET_PANCREAS",
            ],
            "source_cohort": [
                "GSE30929_SINGER_2007_LPS",
                "GSE75885_DELESPAUL_2017",
                "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY",
                "GSE118014_ALVAREZ_2018",
                "GSE98894_ALVAREZ_2018_NET",
            ],
            "source_project": ["GEO", "GEO", "Treehouse", "GEO", "GEO"],
            "source_version": ["v1"] * 5,
            "tumor_origin": ["primary"] * 5,
            "metastasis_site": [pd.NA] * 5,
            "n_reference_genes": [10, 11, 12, 13, 14],
            "n_reference_samples": [5, 6, 7, 8, 9],
            "processing_pipeline": ["rnaseq"] * 5,
            "notes": [""] * 5,
            "selected": [False, False, True, False, True],
        }
    )
    expression._reference_summary_availability_table.cache_clear()
    expression._reference_summary_source_table.cache_clear()
    monkeypatch.setattr(
        expression,
        "get_data",
        lambda name, **kwargs: (
            sidecar
            if name == "cancer-reference-expression-availability"
            else (_ for _ in ()).throw(AssertionError(f"unexpected dataset load: {name}"))
        ),
    )
    monkeypatch.setattr(
        expression,
        "_selected_expression_source_metadata",
        lambda code, *, source_cohort=None: {
            "source_cohort": source_cohort,
            "source_type": "geo",
            "unit": "TPM",
            "source_scale_class": "linear_rnaseq_tpm",
            "linear_tpm_comparable": True,
            "tpm_proxy": False,
        },
    )

    availability = expression.cancer_reference_expression_availability(
        ["SARC_DDLPS", "NET_PANCREAS"],
        reference_source="summary_rows_all",
        sample_qc="all",
        all_sources=True,
    )

    assert len(availability) == 5
    assert set(zip(availability["cancer_code"], availability["source_cohort"])) == set(
        zip(sidecar["cancer_code"], sidecar["source_cohort"])
    )
    assert availability["available"].all()
    assert availability["n_samples"].tolist() == availability["n_reference_samples"].tolist()

    filtered = expression.cancer_reference_expression_availability(
        "NET_PANCREAS",
        reference_source="summary_rows_all",
        sample_qc="all",
        source_cohort="GSE98894_ALVAREZ_2018_NET",
        all_sources=True,
    )
    assert filtered["source_cohort"].tolist() == ["GSE98894_ALVAREZ_2018_NET"]
    assert filtered["n_reference_samples"].tolist() == [9]

    selected = expression.cancer_reference_expression_availability(
        "NET_PANCREAS",
        reference_source="summary_rows",
        sample_qc="all",
    )
    assert selected["source_cohort"].tolist() == ["GSE98894_ALVAREZ_2018_NET"]
    assert selected["n_reference_samples"].tolist() == [9]
    expression._reference_summary_availability_table.cache_clear()
    expression._reference_summary_source_table.cache_clear()


def test_reference_availability_all_sources_preserves_aggregate_requests(monkeypatch):
    sidecar = pd.DataFrame(
        {
            "cancer_code": ["X", "X", "Y"],
            "source_cohort": ["X_ONE", "X_TWO", "Y_ONE"],
            "n_reference_genes": [2, 3, 4],
            "n_reference_samples": [5, 6, 7],
        }
    )
    expression._reference_summary_availability_table.cache_clear()
    monkeypatch.setattr(expression, "get_data", lambda *args, **kwargs: sidecar)
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(expression, "cohort_aggregates", lambda: {"AGG": ["X", "Y"]})
    monkeypatch.setattr(
        expression,
        "_selected_expression_source_metadata",
        lambda code, *, source_cohort=None: {
            "source_cohort": source_cohort,
            "unit": "TPM",
            "source_scale_class": "linear_rnaseq_tpm",
            "linear_tpm_comparable": True,
            "tpm_proxy": False,
        },
    )

    availability = expression.cancer_reference_expression_availability(
        "AGG",
        reference_source="summary_rows_all",
        sample_qc="all",
        all_sources=True,
    )

    assert availability["requested_code"].unique().tolist() == ["AGG"]
    assert set(availability["cancer_code"]) == {"X", "Y"}
    assert set(availability["request_kind"]) == {"aggregate_member"}
    assert len(availability) == 3
    expression._reference_summary_availability_table.cache_clear()


def test_reference_availability_all_sources_requires_source_union_mode():
    with pytest.raises(ValueError, match="all_sources=True requires"):
        expression.cancer_reference_expression_availability(all_sources=True)


def test_reference_availability_sources_are_registry_backed_and_filterable():
    sidecar = expression._reference_summary_availability_table()
    assert set(sidecar["source_cohort"].astype(str)) <= set(
        expression.cohort_registry_df()["cohort_id"]
    )

    merkel = expression.cancer_reference_expression_availability(
        "NEC_MERKEL",
        reference_source="summary_rows_all",
        sample_qc="all",
        source_kind="geo",
        all_sources=True,
    )
    assert merkel["source_cohort"].tolist() == ["GSE235092_MERKEL_2024"]
    assert merkel["source_type"].tolist() == ["geo-matrix"]


def test_artifact_sources_are_registry_backed_with_effective_sample_counts():
    registry_ids = set(expression.cohort_registry_df()["cohort_id"].astype(str))
    matrix_registry = expression.source_matrices.registry()
    matrix_source_ids = set(matrix_registry["source_cohort"].astype(str))
    assert matrix_source_ids <= registry_ids

    all_availability = expression.cancer_reference_expression_availability(
        matrix_registry["cancer_code"].astype(str).tolist(),
        reference_source="artifact",
        sample_qc="artifact",
    )
    assert len(all_availability) == len(matrix_registry)
    assert (
        all_availability[
            [
                "source_cohort",
                "source_project",
                "source_type",
                "source_unit",
                "processing_pipeline",
                "n_samples",
            ]
        ]
        .notna()
        .all()
        .all()
    )
    assert "unknown" not in set(all_availability["source_scale_class"])

    availability = expression.cancer_reference_expression_availability(
        ["SARC_ESS_HG", "SARC_ESS_LG"],
        reference_source="artifact",
        sample_qc="artifact",
    ).set_index("cancer_code")

    assert set(availability["source_cohort"]) == {"GSE85383_YOSHIDA_2017_ESS"}
    assert availability["source_project"].tolist() == ["GEO", "GEO"]
    assert availability["source_type"].tolist() == ["geo-microarray", "geo-microarray"]
    assert availability["source_unit"].tolist() == ["TPM proxy", "TPM proxy"]
    assert availability["source_scale_class"].tolist() == [
        "microarray_tpm_proxy",
        "microarray_tpm_proxy",
    ]
    assert not availability["linear_tpm_comparable"].any()
    assert availability["tumor_origin"].tolist() == ["primary", "primary"]
    assert availability["n_reference_samples"].tolist() == [4, 9]
    assert availability["n_samples"].tolist() == [4, 9]
    assert availability["processing_pipeline"].notna().all()


def test_artifact_build_metadata_uses_loader_source_cohort_identity():
    matrix_registry = expression.source_matrices.registry()
    codes = matrix_registry["cancer_code"].astype(str).tolist()
    metadata = expression.expression_artifact_build_metadata(
        codes, auto_fetch=False, on_missing="raise"
    )
    availability = expression.cancer_reference_expression_availability(
        codes,
        reference_source="artifact",
        sample_qc="artifact",
    )

    comparison = availability.loc[
        availability["available"], ["cancer_code", "source_cohort"]
    ].merge(
        metadata[["cancer_code", "source_cohort", "build_source_cohort"]],
        on="cancer_code",
        suffixes=("_loader", "_metadata"),
        validate="one_to_one",
    )
    assert len(comparison) == len(matrix_registry)
    assert (comparison["source_cohort_loader"] == comparison["source_cohort_metadata"]).all()

    by_code = comparison.set_index("cancer_code")
    assert by_code.loc["LUAD", "source_cohort_metadata"] == "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES"
    assert by_code.loc["LUAD", "build_source_cohort"] == "TREEHOUSE_POLYA_25_01"
    assert by_code.loc["CML", "source_cohort_metadata"] == "GSE100026_DING_2017"
    assert by_code.loc["CML", "build_source_cohort"] == "GEO_HEME_2022"


def test_artifact_reference_uses_selected_build_sample_count(monkeypatch):
    metadata = pd.DataFrame(
        {
            "cancer_code": ["MCL"],
            "sample_qc_effective": ["pass_or_warn"],
            "n_cohort_samples": [10],
        }
    )
    percentiles = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "p25": [1.0],
            "p50": [2.0],
            "p75": [3.0],
        }
    )
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["MCL"])
    monkeypatch.setattr(expression, "expression_artifact_build_metadata", lambda **kwargs: metadata)
    monkeypatch.setattr(
        expression,
        "cohort_gene_percentiles",
        lambda *args, **kwargs: percentiles.copy(),
    )

    availability = expression.cancer_reference_expression_availability(
        "MCL",
        reference_source="artifact",
        sample_qc="artifact",
    )
    reference = expression.cancer_reference_expression(
        "MCL",
        reference_source="artifact",
        sample_qc="artifact",
    )

    assert expression.source_matrices.cohort_info("MCL")["n_samples"] == 51
    assert availability.loc[0, "n_reference_samples"] == 10
    assert availability.loc[0, "n_samples"] == 10
    assert reference["n_reference_samples"].tolist() == [10]
    assert reference["n_samples"].tolist() == [10]


@pytest.mark.parametrize("invalid_count", [-1, 1.5])
def test_artifact_build_metadata_rejects_invalid_sample_counts(invalid_count):
    metadata = pd.DataFrame({"cancer_code": ["X"], "n_cohort_samples": [invalid_count]})

    with pytest.raises(ValueError, match="invalid sample count for X"):
        expression._artifact_build_metadata_sample_counts(metadata)


@pytest.mark.parametrize(
    ("policies", "expected"),
    [
        ({}, (None, "artifact_build_metadata_missing")),
        ({"X": ("all", "pass")}, (None, "artifact_sample_qc_conflict")),
        ({"X": ("unknown",)}, (None, "artifact_sample_qc_invalid")),
        ({"X": ("pass_or_warn",)}, ("pass_or_warn", "")),
    ],
)
def test_artifact_effective_sample_qc_requires_one_valid_policy(policies, expected):
    assert expression._artifact_effective_sample_qc("X", policies) == expected


def test_explicit_source_metadata_does_not_fall_back_to_another_source(monkeypatch):
    from types import SimpleNamespace

    from oncoref import expression_registry

    monkeypatch.setattr(
        expression_registry,
        "sources_for_cancer_code",
        lambda code: [
            SimpleNamespace(
                source_cohort="SELECTED_SOURCE",
                source_type="gdc",
                unit="TPM",
                special_handling="",
                project_id="TCGA-X",
                accession=None,
            )
        ],
    )

    metadata = expression._selected_expression_source_metadata(
        "X", source_cohort="EXPLICIT_OTHER_SOURCE"
    )

    assert metadata["source_cohort"] == "EXPLICIT_OTHER_SOURCE"
    assert metadata["source_type"] is None
    assert metadata["source_project"] is None
    assert metadata["source_scale_class"] == "unknown"
    assert not metadata["linear_tpm_comparable"]


def test_reference_summary_source_labels_canonicalize_treehouse_tcga_cohorts():
    raw = pd.DataFrame(
        {
            "cancer_code": ["SARC_DDLPS", "SARC_WDLPS", "SARC_PLEOLPS", "LUAD"],
            "source_cohort": [
                "TREEHOUSE_POLYA_25_01_TCGA_SUBSET",
                "TREEHOUSE_POLYA_25_01_TCGA_SUBSET",
                "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES",
                "TREEHOUSE_POLYA_25_01_TCGA_SUBSET",
            ],
        }
    )
    canonical = expression._canonical_reference_summary_source_labels(raw)

    assert canonical["source_cohort"].tolist() == [
        "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY",
        "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY",
        "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY",
        "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES",
    ]
    assert raw["source_cohort"].tolist()[2] == "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES"


def test_legacy_treehouse_tcga_filter_is_exact_and_excludes_derived_cohorts():
    table = pd.DataFrame(
        {
            "cancer_code": ["LUAD", "SARC_DDLPS", "UCEC_POLE"],
            "source_cohort": [
                "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES",
                "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY",
                "TREEHOUSE_POLYA_25_01_TCGA_UCEC_SUBTYPE",
            ],
        }
    )
    with pytest.warns(DeprecationWarning, match="TCGA_SUBSET"):
        filters = expression._normalize_source_cohort_filter_values(
            "TREEHOUSE_POLYA_25_01_TCGA_SUBSET"
        )
    filtered = expression._filter_reference_summary_sources(table, source_cohorts=filters)

    assert filtered["source_cohort"].tolist() == ["TREEHOUSE_POLYA_25_01_TCGA_SAMPLES"]


def test_cancer_reference_expression_summary_rows_all_pool(monkeypatch):
    expression._reference_summary_source_table.cache_clear()
    summary = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E1"],
            "Symbol": ["A", "A"],
            "cancer_code": pd.Categorical(["X", "X"], categories=["X", "UNUSED_CODE"]),
            "source_cohort": pd.Categorical(
                ["SRC1", "SRC2"], categories=["SRC1", "SRC2", "UNUSED_SOURCE"]
            ),
            "source_project": ["P1", "P2"],
            "source_version": ["v1", "v2"],
            "TPM_median": [10.0, 20.0],
            "TPM_q1": [9.0, 19.0],
            "TPM_q3": [11.0, 21.0],
            "TPM_clean_median": [1.0, 2.0],
            "TPM_clean_q1": [0.9, 1.9],
            "TPM_clean_q3": [1.1, 2.1],
            "n_samples": [2, 6],
            "n_detected": [2, 5],
            "processing_pipeline": ["rna_seq", "rna_seq"],
            "notes": ["a", "b"],
            "tumor_origin": ["primary", "primary"],
            "metastasis_site": [pd.NA, pd.NA],
        }
    )
    summary_availability = pd.DataFrame(
        {
            "cancer_code": ["X", "X"],
            "source_cohort": ["SRC1", "SRC2"],
            "source_project": ["P1", "P2"],
            "source_version": ["v1", "v2"],
            "n_reference_genes": [1, 1],
            "n_reference_samples": [2, 6],
            "processing_pipeline": ["rna_seq", "rna_seq"],
            "selected": [False, True],
        }
    )
    _mock_reference_summary_shards(monkeypatch, summary)
    monkeypatch.setattr(
        expression,
        "_reference_summary_availability_table",
        lambda: summary_availability,
    )
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code, **k: str(code).upper())

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        pooled = expression.cancer_reference_expression(
            "x",
            normalize="tpm",
            reference_source="summary_rows_all",
            sample_qc="all",
            pool=True,
        )

    assert len(pooled) == 1
    row = pooled.iloc[0]
    assert row["source_cohort"] == "POOLED"
    assert row["expression"] == pytest.approx((10.0 * 2 + 20.0 * 6) / 8)
    assert pd.isna(row["q1"]) and pd.isna(row["q3"])
    assert row["n_reference_samples"] == pytest.approx(8)
    assert row["processing_pipeline"] == "pooled_n_weighted"
    assert row["sample_qc_effective"] == "all"
    assert row["Proteoform_ID"] == "E1"
    assert row["Member_Ensembl_Gene_IDs"] == "E1"

    compact = expression.cancer_reference_expression(
        "x",
        normalize="tpm",
        reference_source="summary_rows_all",
        sample_qc="all",
        pool=True,
        include_provenance=False,
    )
    assert len(compact) == 1
    assert compact.loc[0, "source_cohort"] == "POOLED"
    assert compact.loc[0, "expression"] == pytest.approx((10.0 * 2 + 20.0 * 6) / 8)
    assert compact.loc[0, "n_reference_samples"] == pytest.approx(8)
    assert compact.loc[0, "Proteoform_ID"] == "E1"
    assert compact.loc[0, "Member_Ensembl_Gene_IDs"] == "E1"
    assert "processing_pipeline" not in compact.columns

    selected_source = expression.cancer_reference_expression(
        "x",
        normalize="tpm",
        reference_source="summary_rows_all",
        sample_qc="all",
        source_cohort="SRC2",
        pool=True,
        on_missing="empty",
    )
    assert len(selected_source) == 1
    assert selected_source.loc[0, "source_cohort"] == "POOLED"
    assert selected_source.loc[0, "expression"] == pytest.approx(20.0)
    availability = selected_source.attrs["availability"]
    assert len(availability) == 1
    assert availability[0]["available"] is True
    assert availability[0]["source_cohort"] == "SRC2"

    with pytest.raises(ValueError, match='requires sample_qc="all"'):
        expression.cancer_reference_expression("x", reference_source="summary_rows_all")

    expression._reference_summary_source_table.cache_clear()


def test_cancer_reference_expression_summary_rows_qc_filtered_recomputes(monkeypatch):
    stats = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "p25": [1.0],
            "p50": [2.0],
            "p75": [3.0],
        }
    )
    seen = {}
    expression._reference_summary_source_table.cache_clear()
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: [])
    monkeypatch.setattr(expression.source_matrices, "available_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code, **k: str(code).upper())
    monkeypatch.setattr(
        expression,
        "cohort_stats",
        lambda code, **k: seen.update(code=code, **k) or stats.copy(),
    )
    monkeypatch.setattr(
        expression,
        "_selected_expression_source_metadata",
        lambda code: {
            "source_cohort": "SRC_X",
            "source_project": "TCGA",
            "source_version": "v1",
            "source_type": "gdc",
            "unit": "TPM",
            "source_scale_class": "linear_rnaseq_tpm",
            "linear_tpm_comparable": True,
            "tumor_origin": None,
            "metastasis_site": None,
            "n_reference_genes": None,
            "n_reference_samples": 7,
        },
    )

    out = expression.cancer_reference_expression(
        "x",
        reference_source="summary_rows",
        sample_qc="pass_or_warn",
        auto_fetch=True,
        normalize="tpm_clean",
    )

    assert seen == {
        "code": "X",
        "normalize": "tpm_clean",
        "auto_fetch": True,
        "sample_qc": "pass_or_warn",
    }
    assert out["reference_method"].tolist() == ["source_matrix_stats"]
    assert out["sample_qc"].tolist() == ["pass_or_warn"]
    assert out["source_cohort"].tolist() == ["SRC_X"]
    assert out["n_reference_samples"].tolist() == [7]
    assert out["expression"].tolist() == [2.0]


def test_cancer_reference_expression_summary_rows_reports_no_samples_after_qc(monkeypatch):
    stats = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "p25": [1.0],
            "p50": [2.0],
            "p75": [3.0],
        }
    )
    qc = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "sample_qc_status": ["warn", "warn"],
        }
    )
    seen = {}
    expression._source_matrix_effective_sample_count.cache_clear()
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: [])
    monkeypatch.setattr(expression.source_matrices, "available_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code, **k: str(code).upper())
    monkeypatch.setattr(expression, "sample_expression_qc", lambda *a, **k: qc.copy())
    monkeypatch.setattr(
        expression,
        "cohort_stats",
        lambda code, **k: seen.update(code=code, **k) or stats.copy(),
    )

    empty = expression.cancer_reference_expression(
        "x", reference_source="summary_rows", sample_qc="pass", on_missing="empty"
    )

    assert empty.empty
    assert seen == {}
    assert empty.attrs["missing_requests"][0]["missing_reason"] == (
        "no_source_matrix_samples_matching_pass_qc"
    )

    availability = expression.cancer_reference_expression_availability(
        "x", reference_source="summary_rows", sample_qc="pass"
    )
    assert availability["available"].tolist() == [False]
    assert availability["missing_reason"].tolist() == ["no_source_matrix_samples_matching_pass_qc"]

    out = expression.cancer_reference_expression(
        "x", reference_source="summary_rows", sample_qc="pass_or_warn"
    )

    assert seen == {
        "code": "X",
        "normalize": "tpm_clean",
        "auto_fetch": False,
        "sample_qc": "pass_or_warn",
    }
    assert out["expression"].tolist() == [2.0]
    assert out["sample_qc"].tolist() == ["pass_or_warn"]
    expression._source_matrix_effective_sample_count.cache_clear()


def test_cancer_reference_expression_wide_merges_by_gene_id_not_symbol(monkeypatch):
    by_code = {
        "X": pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["E1", "E2"],
                "Symbol": ["A", "B"],
                "p25": [1.0, 2.0],
                "p50": [3.0, 4.0],
                "p75": [5.0, 6.0],
            }
        ),
        "Y": pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["E1", "E3"],
                "Symbol": ["A_OLD", "C"],
                "p25": [5.0, 8.0],
                "p50": [7.0, 9.0],
                "p75": [9.0, 10.0],
            }
        ),
    }
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X", "Y"])
    _mock_missing_artifact_build_metadata(monkeypatch)
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(
        expression, "cohort_gene_percentiles", lambda code, *a, **k: by_code[code].copy()
    )

    wide = expression.cancer_reference_expression(["x", "y"], format="wide")
    e1 = wide[wide["Ensembl_Gene_ID"] == "E1"]
    assert len(e1) == 1
    assert e1["Symbol"].iloc[0] == "A"
    assert e1["X_TPM_clean"].iloc[0] == pytest.approx(3.0)
    assert e1["Y_TPM_clean"].iloc[0] == pytest.approx(7.0)


def test_proteoform_named_accessors_delegate_with_proteoform_true(monkeypatch):
    # The proteoform_* named accessors are thin wrappers that set proteoform=True on
    # their gene-level base, threading scope/statistic/etc. through.
    seen = {}
    empty = pd.DataFrame()

    monkeypatch.setattr(
        expression, "per_sample_expression", lambda c, **k: seen.update(ps=k) or empty
    )
    monkeypatch.setattr(
        expression, "cohort_mean_expression", lambda c, **k: seen.update(mean=k) or empty
    )
    monkeypatch.setattr(
        expression, "cohort_gene_percentiles", lambda c, **k: seen.update(pct=k) or empty
    )
    monkeypatch.setattr(
        expression, "within_sample_top_fraction", lambda c, **k: seen.update(ws=k) or empty
    )

    expression.proteoform_per_sample_expression("X", scope="genome")
    expression.proteoform_cohort_mean_expression("X", statistic="median", scope="genome")
    expression.proteoform_cohort_percentiles("X", as_tpm=False)
    expression.proteoform_within_sample_top_fraction("X", threshold=0.9)

    assert seen["ps"]["proteoform"] is True and seen["ps"]["scope"] == "genome"
    assert seen["mean"]["proteoform"] is True
    assert seen["mean"]["statistic"] == "median" and seen["mean"]["scope"] == "genome"
    assert seen["pct"]["proteoform"] is True and seen["pct"]["as_tpm"] is False
    assert seen["ws"]["proteoform"] is True and seen["ws"]["threshold"] == 0.9


def test_gene_named_accessors_are_gene_level(monkeypatch):
    # The symmetric gene_* accessors delegate to the gene-level base (proteoform not set,
    # or False), exposing only gene-relevant params.
    seen = {}
    empty = pd.DataFrame()

    monkeypatch.setattr(
        expression, "per_sample_expression", lambda c, **k: seen.update(ps=k) or empty
    )
    monkeypatch.setattr(
        expression, "cohort_mean_expression", lambda c, **k: seen.update(mean=k) or empty
    )
    monkeypatch.setattr(
        expression, "cohort_gene_percentiles", lambda c, **k: seen.update(pct=k) or empty
    )
    monkeypatch.setattr(
        expression, "within_sample_top_fraction", lambda c, **k: seen.update(ws=k) or empty
    )

    expression.gene_per_sample_expression("X")
    expression.gene_cohort_mean_expression("X", statistic="median")
    expression.gene_cohort_percentiles("X", as_tpm=False)
    expression.gene_within_sample_top_fraction("X", threshold=0.9)

    assert seen["ps"].get("proteoform", False) is False  # gene level
    assert seen["mean"].get("proteoform", False) is False and seen["mean"]["statistic"] == "median"
    assert seen["pct"].get("proteoform", False) is False and seen["pct"]["as_tpm"] is False
    assert seen["ws"].get("proteoform", False) is False and seen["ws"]["threshold"] == 0.9


def test_cohort_stats_full_suite(monkeypatch):
    fixture = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "s1": [10.0],
            "s2": [20.0],
            "s3": [30.0],
            "s4": [40.0],
        }
    )
    monkeypatch.setattr(expression, "per_sample_expression", lambda *a, **k: fixture.copy())
    s = expression.cohort_stats("X")
    assert set(s.columns) >= {
        "mean",
        "std",
        "min",
        "p1",
        "p5",
        "p10",
        "p15",
        "p20",
        "p25",
        "p50",
        "p75",
        "p80",
        "p85",
        "p90",
        "p95",
        "p99",
        "max",
    }
    row = s.iloc[0]
    assert row["mean"] == pytest.approx(25.0)
    assert row["std"] == pytest.approx(np.std([10, 20, 30, 40]))
    assert row["min"] == 10.0 and row["max"] == 40.0
    assert row["p50"] == pytest.approx(25.0)
    assert row["p25"] == pytest.approx(17.5) and row["p75"] == pytest.approx(32.5)
    assert row["p90"] == pytest.approx(37.0)


def test_cohort_stats_named_accessors_delegate(monkeypatch):
    seen = {}
    fixture = pd.DataFrame({"Ensembl_Gene_ID": ["E1"], "Symbol": ["A"], "s1": [1.0], "s2": [2.0]})
    monkeypatch.setattr(
        expression, "per_sample_expression", lambda c, **k: seen.update(k) or fixture.copy()
    )
    expression.gene_cohort_stats("X")
    assert seen.get("proteoform", False) is False
    expression.proteoform_cohort_stats("X", scope="genome")
    assert seen["proteoform"] is True and seen["scope"] == "genome"


def _pan_cancer_fixture():
    return pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000001.5", "ENSG00000002"],
            "Symbol": ["GENE1", "GENE2"],
            "nTPM_liver": [3.0, 7.0],
            "FPKM_LUAD": [2.0, 8.0],
            "FPKM_BLCA": [5.0, 5.0],
        }
    )


def test_pan_cancer_expression_converts_fpkm_to_tpm(monkeypatch):
    _mock_pan_cancer_data_without_computed_aggregates(monkeypatch, _pan_cancer_fixture())
    out = expression.pan_cancer_expression()
    # FPKM_<CODE> tumor columns become entity-first <CODE>_TPM_raw;
    # HPA nTPM columns become <tissue>_nTPM_raw. Clean companions are default.
    assert "LUAD_FPKM_raw" in out.columns and "FPKM_LUAD" not in out.columns
    assert "LUAD_TPM_raw" in out.columns
    assert "liver_nTPM_raw" in out.columns
    assert "LUAD_TPM_clean" in out.columns and "liver_nTPM_clean" in out.columns
    assert "TPM_LUAD" not in out.columns and "nTPM_liver" not in out.columns
    assert out["LUAD_FPKM_raw"].tolist() == pytest.approx([2.0, 8.0])
    # Each TCGA column is rescaled to sum 1e6: FPKM_LUAD [2,8] -> [200000, 800000].
    assert out["LUAD_TPM_raw"].tolist() == pytest.approx([200000.0, 800000.0])
    assert out["BLCA_TPM_raw"].sum() == pytest.approx(1e6)


def test_pan_cancer_expression_preserves_unmeasured_source_values(monkeypatch):
    fixture = _pan_cancer_fixture()
    fixture.loc[0, "FPKM_LUAD"] = np.nan
    _mock_pan_cancer_data_without_computed_aggregates(monkeypatch, fixture)

    out = expression.pan_cancer_expression()

    assert pd.isna(out.loc[0, "LUAD_TPM_raw"])
    assert pd.isna(out.loc[0, "LUAD_TPM_clean"])
    assert out.loc[1, "LUAD_TPM_clean"] == pytest.approx(normalization.BIOLOGICAL_FRACTION * 1e6)


def test_pan_cancer_expression_raw_only(monkeypatch):
    _mock_pan_cancer_data_without_computed_aggregates(monkeypatch, _pan_cancer_fixture())
    out = expression.pan_cancer_expression(normalize=None)
    assert "LUAD_FPKM_raw" in out.columns
    assert "LUAD_TPM_raw" in out.columns
    assert "LUAD_TPM_clean" not in out.columns
    assert out["LUAD_FPKM_raw"].tolist() == pytest.approx([2.0, 8.0])
    assert out["LUAD_TPM_raw"].tolist() == pytest.approx([200000.0, 800000.0])


def test_pan_cancer_expression_pirlygenes_column_style(monkeypatch):
    _mock_pan_cancer_data_without_computed_aggregates(monkeypatch, _pan_cancer_fixture())
    out = expression.pan_cancer_expression(normalize="tpm", column_style="pirlygenes")

    assert {"liver_nTPM", "LUAD_FPKM", "LUAD_TPM"} <= set(out.columns)
    assert not any(c.endswith("_raw") for c in out.columns)
    assert out["LUAD_FPKM"].tolist() == pytest.approx([2.0, 8.0])
    assert out["LUAD_TPM"].tolist() == pytest.approx([200000.0, 800000.0])
    assert out.attrs["oncoref"]["column_style"] == "pirlygenes"


def test_pan_cancer_expression_adds_computed_aggregate_tpm_columns(monkeypatch):
    source_table = pd.DataFrame(
        {
            "cancer_code": [
                "NET_PANCREAS",
                "COAD",
                "READ",
                "LUAD",
                "LUSC",
                "CHOL",
                "ADCC",
            ],
            "source_cohort": [
                "NET_SRC",
                "COAD_SRC",
                "READ_SRC",
                "LUAD_SRC",
                "LUSC_SRC",
                "CHOL_SRC",
                "ADCC_SRC",
            ],
            "selected": [True, True, True, True, True, True, True],
        }
    )
    summary_rows = []
    values = {
        "NET_PANCREAS": ([7.0, 9.0], 5, "NET_SRC"),
        "COAD": ([10.0, 20.0], 2, "COAD_SRC"),
        "READ": ([30.0, 40.0], 6, "READ_SRC"),
        "LUAD": ([100.0, 200.0], 1, "LUAD_SRC"),
        "LUSC": ([300.0, 400.0], 3, "LUSC_SRC"),
        "CHOL": ([11.0, 12.0], 4, "CHOL_SRC"),
        "ADCC": ([21.0, 22.0], 8, "ADCC_SRC"),
    }
    for code, (exprs, n_samples, source) in values.items():
        for gene_id, symbol, expr in zip(
            ["ENSG00000001", "ENSG00000002"], ["GENE1", "GENE2"], exprs
        ):
            summary_rows.append(
                {
                    "Ensembl_Gene_ID": gene_id,
                    "Symbol": symbol,
                    "cancer_code": code,
                    "source_cohort": source,
                    "TPM_median": expr,
                    "n_samples": n_samples,
                }
            )
    summary = pd.DataFrame(summary_rows)
    members = {
        "NET": ("NET_PANCREAS",),
        "CRC": ("COAD", "READ"),
        "NSCLC": ("LUAD", "LUSC"),
        "BTC": ("CHOL",),
        "SGC": ("ADCC",),
    }

    monkeypatch.setattr(expression, "get_data", lambda *args, **kwargs: _pan_cancer_fixture())
    monkeypatch.setattr(
        expression, "_computed_expression_reference_members", lambda code: members.get(code, ())
    )
    monkeypatch.setattr(expression, "_reference_summary_source_table", lambda: source_table)
    _mock_reference_summary_shards(monkeypatch, summary)

    out = expression.pan_cancer_expression(normalize="tpm", column_style="pirlygenes")

    assert {"NET_TPM", "CRC_TPM", "NSCLC_TPM", "BTC_TPM", "SGC_TPM"} <= set(out.columns)
    assert out["NET_TPM"].tolist() == pytest.approx([7.0, 9.0])
    assert out["CRC_TPM"].tolist() == pytest.approx([25.0, 35.0])
    assert out["NSCLC_TPM"].tolist() == pytest.approx([250.0, 350.0])
    assert out["BTC_TPM"].tolist() == pytest.approx([11.0, 12.0])
    assert out["SGC_TPM"].tolist() == pytest.approx([21.0, 22.0])
    assert out.attrs["oncoref"]["computed_aggregate_columns"] == (
        "NET_TPM_raw",
        "CRC_TPM_raw",
        "NSCLC_TPM_raw",
        "BTC_TPM_raw",
        "SGC_TPM_raw",
    )


def test_pan_cancer_expression_to_tpm_legacy_keyword(monkeypatch):
    _mock_pan_cancer_data_without_computed_aggregates(monkeypatch, _pan_cancer_fixture())
    out = expression.pan_cancer_expression(genes=["ENSG00000001"], to_tpm=True)

    assert out["Symbol"].tolist() == ["GENE1"]
    assert {"liver_nTPM", "LUAD_FPKM", "LUAD_TPM"} <= set(out.columns)
    assert "LUAD_TPM_clean" not in out.columns
    assert out["LUAD_TPM"].iloc[0] == pytest.approx(200000.0)


def test_pan_cancer_expression_empty_gene_filter_preserves_schema(monkeypatch):
    _mock_pan_cancer_data_without_computed_aggregates(monkeypatch, _pan_cancer_fixture())
    empty = expression.pan_cancer_expression(genes=[], normalize="tpm", column_style="pirlygenes")
    full = expression.pan_cancer_expression(normalize="tpm", column_style="pirlygenes")

    assert empty.empty
    assert list(empty.columns) == list(full.columns)
    assert empty.attrs["oncoref"]["dataset"] == "pan-cancer-expression"


def test_pan_cancer_expression_bad_column_style(monkeypatch):
    _mock_pan_cancer_data_without_computed_aggregates(monkeypatch, _pan_cancer_fixture())
    with pytest.raises(ValueError, match="column_style"):
        expression.pan_cancer_expression(column_style="source")


def test_pan_cancer_expression_accepts_clean_tpm_alias(monkeypatch):
    _mock_pan_cancer_data_without_computed_aggregates(monkeypatch, _pan_cancer_fixture())
    canonical = expression.pan_cancer_expression(normalize="tpm_clean")
    alias = expression.pan_cancer_expression(normalize="clean_tpm")
    assert "LUAD_TPM_clean" in alias.columns
    assert alias["LUAD_TPM_clean"].tolist() == pytest.approx(canonical["LUAD_TPM_clean"].tolist())


def test_pan_cancer_expression_housekeeping_mode(monkeypatch):
    import oncoref.gene_families as gf

    _mock_pan_cancer_data_without_computed_aggregates(monkeypatch, _pan_cancer_fixture())
    monkeypatch.setattr(
        gf, "clean_tpm_biological_housekeeping_gene_ids", lambda: frozenset({"ENSG00000001"})
    )
    out = expression.pan_cancer_expression(normalize=["tpm_clean", "hk", "percentile"])
    assert "LUAD_TPM_hk" in out.columns
    assert "liver_nTPM_hk" in out.columns
    assert "LUAD_TPM_percentile" in out.columns


def test_pan_cancer_expression_log_modes(monkeypatch):
    _mock_pan_cancer_data_without_computed_aggregates(monkeypatch, _pan_cancer_fixture())
    raw_logged = expression.pan_cancer_expression(normalize="tpm_log1p")
    clean_logged = expression.pan_cancer_expression(normalize="tpm_clean_log1p")
    assert "LUAD_TPM_raw_log1p" in raw_logged.columns
    assert "LUAD_TPM_clean_log1p" not in raw_logged.columns
    assert raw_logged["LUAD_TPM_raw_log1p"].iloc[0] == pytest.approx(
        np.log1p(raw_logged["LUAD_TPM_raw"].iloc[0])
    )
    assert "LUAD_TPM_clean_log1p" in clean_logged.columns


def test_pan_cancer_expression_gene_filter(monkeypatch):
    _mock_pan_cancer_data_without_computed_aggregates(monkeypatch, _pan_cancer_fixture())
    # Filter by symbol, and by unversioned Ensembl id (fixture id is versioned).
    by_symbol = expression.pan_cancer_expression(genes="GENE2")
    assert by_symbol["Symbol"].tolist() == ["GENE2"]
    by_id = expression.pan_cancer_expression(genes=["ENSG00000001"])
    assert by_id["Symbol"].tolist() == ["GENE1"]
    # Conversion still reflects the cohort-wide scaling computed before filtering.
    assert by_id["LUAD_TPM_raw"].iloc[0] == pytest.approx(200000.0)


def test_pooled_cohort_stats_availability_and_heterogeneity(monkeypatch):
    # Two cohorts with overlapping but ragged gene sets and very different sizes.
    # BIG measures E1+E2 (3 samples), SMALL measures E2+E3 (1 sample). E2 is shared.
    big = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E2"],
            "Symbol": ["A", "B"],
            "s1": [10.0, 0.0],
            "s2": [20.0, 0.0],
            "s3": [30.0, 6.0],
        }
    )
    small = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E2", "E3"],
            "Symbol": ["B", "C"],
            "x1": [100.0, 4.0],
        }
    )
    frames = {"BIG": big, "SMALL": small}
    monkeypatch.setattr(expression, "per_sample_expression", lambda code, **k: frames[code].copy())
    monkeypatch.setattr(expression, "_resolve_cancer_types", lambda ct, **k: list(ct))

    out = expression.pooled_cohort_stats(["BIG", "SMALL"]).set_index("Ensembl_Gene_ID")

    # n_samples is the constant pooled width; n_available is the honest per-gene
    # denominator; n_cohorts counts measuring cohorts.
    assert (out["n_samples"] == 4).all()
    assert out.loc["E1", "n_available"] == 3 and out.loc["E1", "n_cohorts"] == 1
    assert out.loc["E2", "n_available"] == 4 and out.loc["E2", "n_cohorts"] == 2
    assert out.loc["E3", "n_available"] == 1 and out.loc["E3", "n_cohorts"] == 1
    # n_detected ignores measured-but-zero: E2 is 0 in BIG's s1/s2, >0 in s3 + SMALL.
    assert out.loc["E2", "n_detected"] == 2

    # E2 sample-pooled mean is (0+0+6+100)/4 = 26.5; the balanced (equal-cohort) mean
    # is mean(BIG_mean=2.0, SMALL_mean=100.0) = 51.0 — the big cohort no longer
    # dominates. std_between captures the cross-cohort spread (>0 for E2).
    assert out.loc["E2", "mean"] == pytest.approx(26.5)
    assert out.loc["E2", "balanced_mean"] == pytest.approx(51.0)
    assert out.loc["E2", "std_between"] == pytest.approx(np.std([2.0, 100.0]))
    # A single-cohort gene has no between-cohort spread.
    assert np.isnan(out.loc["E1", "std_between"])
    # std is NaN for a gene with <2 measured samples (E3: one sample in SMALL); a
    # multi-sample gene gets a real std.
    assert np.isnan(out.loc["E3", "std"])
    assert out.loc["E1", "std"] == pytest.approx(np.std([10.0, 20.0, 30.0]))
    # Off-panel cells are never imputed to zero: E1's max is BIG's max (30), not
    # dragged down by SMALL's missing samples.
    assert out.loc["E1", "max"] == 30.0


def test_pooled_cohort_stats_merges_alt_haplotype_aliases(tmp_path, monkeypatch):
    # End-to-end #465-class fix: per_sample_expression now returns the dense canonical
    # space, so when cohort A carries an alt-haplotype id and cohort B its primary,
    # pooling sees ONE canonical gene across both cohorts — not two disjoint sparse rows.
    # Drive the REAL per_sample (normalize="tpm_raw" so values pass through clean_tpm-free
    # for an exact pooled mean); the canonicalization itself lives one layer down now.
    from oncoref.gene_ids import ensembl_id_aliases

    alt, primary = next((a, p) for a, p in ensembl_id_aliases().items() if a != p)
    pa, pb = tmp_path / "A.parquet", tmp_path / "B.parquet"
    pd.DataFrame({"Ensembl_Gene_ID": [alt], "Symbol": ["G"], "s1": [10.0]}).to_parquet(
        pa, index=False
    )
    pd.DataFrame({"Ensembl_Gene_ID": [primary], "Symbol": ["G"], "x1": [20.0]}).to_parquet(
        pb, index=False
    )
    paths = {"A": pa, "B": pb}
    expression._load_per_sample_matrix.cache_clear()
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: paths[code])
    monkeypatch.setattr(expression, "_resolve_cancer_types", lambda ct, **k: list(ct))

    out = expression.pooled_cohort_stats(
        ["A", "B"], normalize="tpm_raw", sample_qc="all"
    ).set_index("Ensembl_Gene_ID")
    assert list(out.index) == [primary]  # one canonical row, the primary id
    assert alt not in set(out.index)
    # both cohorts pooled onto it (mean of the two single-sample cohorts)
    assert out.loc[primary, "n_cohorts"] == 2
    assert out.loc[primary, "n_samples"] == 2
    assert out.loc[primary, "mean"] == pytest.approx(15.0)


def test_pooled_cohort_stats_expands_aggregate_cohorts(monkeypatch):
    # An aggregate code (CRC = COAD + READ) pools its member subtypes, not the
    # rollup label (which has no per-sample matrix of its own).
    served = {
        "COAD": pd.DataFrame({"Ensembl_Gene_ID": ["E1"], "Symbol": ["A"], "c1": [10.0]}),
        "READ": pd.DataFrame({"Ensembl_Gene_ID": ["E1"], "Symbol": ["A"], "r1": [20.0]}),
    }
    monkeypatch.setattr(expression, "per_sample_expression", lambda code, **k: served[code].copy())
    out = expression.pooled_cohort_stats(["CRC"]).set_index("Ensembl_Gene_ID")
    # Both members contributed: 2 cohorts, 2 samples, pooled mean (10+20)/2 = 15.
    assert out.loc["E1", "n_cohorts"] == 2
    assert out.loc["E1", "n_samples"] == 2
    assert out.loc["E1", "mean"] == pytest.approx(15.0)


def test_pooled_cohort_stats_min_cohorts_filter(monkeypatch):
    big = pd.DataFrame({"Ensembl_Gene_ID": ["E1", "E2"], "Symbol": ["A", "B"], "s1": [1.0, 2.0]})
    small = pd.DataFrame({"Ensembl_Gene_ID": ["E2"], "Symbol": ["B"], "x1": [3.0]})
    frames = {"BIG": big, "SMALL": small}
    monkeypatch.setattr(expression, "per_sample_expression", lambda code, **k: frames[code].copy())
    monkeypatch.setattr(expression, "_resolve_cancer_types", lambda ct, **k: list(ct))
    out = expression.pooled_cohort_stats(["BIG", "SMALL"], min_cohorts=2)
    # Only E2 is measured by both cohorts.
    assert out["Ensembl_Gene_ID"].tolist() == ["E2"]


def test_pooled_cohort_stats_requires_cohorts(monkeypatch):
    monkeypatch.setattr(expression, "_resolve_cancer_types", lambda ct, **k: [])
    with pytest.raises(ValueError, match="at least one cancer type"):
        expression.pooled_cohort_stats([])


def test_pooled_cohort_stats_named_accessors_delegate(monkeypatch):
    seen = {}

    def _fake_pool(ct, **k):
        seen.update(k)
        seen["ct"] = ct
        return pd.DataFrame()

    monkeypatch.setattr(expression, "pooled_cohort_stats", _fake_pool)
    expression.gene_pooled_cohort_stats(["X"])
    assert seen.get("proteoform", False) is False
    expression.proteoform_pooled_cohort_stats(["X"], scope="genome")
    assert seen["proteoform"] is True and seen["scope"] == "genome"


def test_cohort_mean_expression_bad_statistic(monkeypatch):
    monkeypatch.setattr(expression, "per_sample_expression", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError, match="statistic must be"):
        expression.cohort_mean_expression("X", statistic="mode")


def test_cohort_mean_expression_reduces_proteoform_frame(monkeypatch):
    # Given a proteoform-level per-sample frame (members already summed), cohort_mean
    # reduces over patients keeping the proteoform key space.
    collapsed = pd.DataFrame(
        {
            "proteoform_key": ["A1/2", "ENSG_B"],
            "Ensembl_Gene_ID": ["ENSG_A1", "ENSG_B"],
            "Symbol": ["A1/2", "B"],
            "proteoform_members": ["A1/A2", "B"],
            "s1": [8.0, 1.0],  # A1+A2 already summed per sample
            "s2": [2.0, 9.0],
        }
    )
    monkeypatch.setattr(expression, "per_sample_expression", lambda *a, **k: collapsed.copy())
    out = expression.cohort_mean_expression("X", statistic="mean", proteoform=True)
    assert "proteoform_key" in out.columns
    by_key = dict(zip(out["proteoform_key"], out["expression"]))
    assert by_key["A1/2"] == pytest.approx(5.0)  # (8 + 2) / 2
    assert by_key["ENSG_B"] == pytest.approx(5.0)  # (1 + 9) / 2, singleton keyed by ENSG
