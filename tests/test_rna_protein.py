# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import oncoref
from oncoref import rna_protein

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_rna_protein_calibration.py"
_SPEC = importlib.util.spec_from_file_location("build_rna_protein_calibration", _SCRIPT_PATH)
builder = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = builder
_SPEC.loader.exec_module(builder)

_FIT_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "fit_rna_protein_calibrations.py"
)
_FIT_SPEC = importlib.util.spec_from_file_location("fit_rna_protein_calibrations", _FIT_SCRIPT_PATH)
fitter = importlib.util.module_from_spec(_FIT_SPEC)
sys.modules[_FIT_SPEC.name] = fitter
_FIT_SPEC.loader.exec_module(fitter)

_HPA_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_rna_protein_hpa_priors.py"
)
_HPA_SPEC = importlib.util.spec_from_file_location("build_rna_protein_hpa_priors", _HPA_SCRIPT_PATH)
hpa_builder = importlib.util.module_from_spec(_HPA_SPEC)
sys.modules[_HPA_SPEC.name] = hpa_builder
_HPA_SPEC.loader.exec_module(hpa_builder)


def test_cptac_source_manifest_is_complete_and_pinned():
    frame = rna_protein.rna_protein_calibration_sources()

    assert tuple(frame.columns) == rna_protein.RNA_PROTEIN_SOURCE_COLUMNS
    assert len(frame) == 20
    assert set(frame["cptac_cohort"]) == set(rna_protein.CPTAC_CALIBRATION_COHORTS)
    modalities = frame.groupby("cptac_cohort")["modality"].apply(set)
    assert all(value == {"rna", "protein"} for value in modalities)
    assert set(frame["source_record"]) == {"10.5281/zenodo.8394329"}
    assert set(frame["license_id"]) == {"CC-BY-4.0"}
    assert set(frame["processing_version"]) == {"1.5.14"}
    assert set(frame["processing_commit"]) == {"2255e5d20cf66ac8072ccfbb66519d4f7e7fb06a"}


def test_cptac_source_manifest_filters_are_exact_and_case_insensitive():
    ucec_rna = rna_protein.rna_protein_calibration_sources(cptac_cohort=" ucec ", modality="RNA")

    assert ucec_rna["source_id"].tolist() == ["CPTAC_BCM_UCEC_RNA"]
    assert rna_protein.rna_protein_calibration_sources(modality="ihc").empty


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda frame: frame.__setitem__("source_id", ["same"] * len(frame)), "source_id"),
        (
            lambda frame: frame.__setitem__(
                "source_scale", ["wrong", *frame["source_scale"].iloc[1:].tolist()]
            ),
            "source_scale",
        ),
        (
            lambda frame: frame.__setitem__(
                "checksum", ["sha256:no", *frame["checksum"].iloc[1:].tolist()]
            ),
            "checksums",
        ),
    ],
)
def test_source_manifest_validation_rejects_integrity_breaks(mutation, match):
    frame = rna_protein.rna_protein_calibration_sources()
    mutation(frame)

    with pytest.raises(ValueError, match=match):
        rna_protein._validate_rna_protein_calibration_sources(frame)


def test_source_manifest_validation_requires_both_modalities():
    frame = rna_protein.rna_protein_calibration_sources()
    frame = frame.loc[frame["source_id"].ne("CPTAC_BCM_UCEC_PROTEIN")]

    with pytest.raises(ValueError, match="one RNA and one protein"):
        rna_protein._validate_rna_protein_calibration_sources(frame)


def test_source_file_verification_checks_size_and_digest(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"matched-source")
    source = pd.Series(
        {
            "source_id": "synthetic",
            "byte_size": path.stat().st_size,
            "checksum": "md5:" + hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest(),
        }
    )

    builder.verify_source_file(path, source)
    source["checksum"] = "md5:" + "0" * 32
    with pytest.raises(ValueError, match="MD5 mismatch"):
        builder.verify_source_file(path, source)


def test_matched_cptac_join_is_canonical_deterministic_and_lossless_for_missingness(
    tmp_path,
):
    tp53 = "ENSG00000141510.17"
    egfr = "ENSG00000146648.16"
    retired_ggnbp2 = "ENSG00000005955.7"
    current_ggnbp2 = "ENSG00000278311.4"
    multi_gene = f"{tp53};{egfr}"
    unknown = "ENSG99999999999.1"
    genes = [tp53, egfr, retired_ggnbp2, current_ggnbp2, multi_gene, unknown]

    rna = pd.DataFrame(
        {
            "S2_T": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "S1": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
            "S3_A": [20.0] * 6,
            "RNA_ONLY_T": [30.0] * 6,
        },
        index=genes,
    )
    protein = pd.DataFrame(
        {
            "S1": [10.0, float("nan"), 50.0, 70.0, 90.0, 110.0],
            "S2_T": [20.0, 40.0, 60.0, 80.0, 100.0, 120.0],
            "PROTEIN_ONLY": [1.0] * 6,
        },
        index=genes,
    )
    rna_path = tmp_path / "rna.tsv"
    protein_path = tmp_path / "protein.tsv"
    rna.to_csv(rna_path, sep="\t")
    protein.to_csv(protein_path, sep="\t")

    first = builder.load_matched_cptac_cohort(rna_path, protein_path)
    second = builder.load_matched_cptac_cohort(rna_path, protein_path)

    pd.testing.assert_frame_equal(first.rna, second.rna)
    pd.testing.assert_frame_equal(first.protein, second.protein)
    assert first.samples["sample_id"].tolist() == ["S1", "S2"]
    assert first.samples["rna_source_sample_id"].tolist() == ["S1", "S2_T"]
    assert first.samples["protein_source_sample_id"].tolist() == ["S1", "S2_T"]
    assert first.genes.to_dict(orient="records") == [
        {"canonical_gene_id": "ENSG00000141510", "source_gene_id": tp53},
        {"canonical_gene_id": "ENSG00000146648", "source_gene_id": egfr},
    ]
    assert pd.isna(first.protein.loc["ENSG00000146648", "S1"])
    assert first.rna.loc["ENSG00000141510", "S1"] == 1.0
    reasons = first.excluded_genes.set_index("source_gene_id")["exclusion_reason"].to_dict()
    assert reasons[multi_gene] == "multi_gene_group"
    assert reasons[unknown] == "unmapped_gene_id"
    assert reasons[retired_ggnbp2] == "duplicate_canonical_gene"
    assert reasons[current_ggnbp2] == "duplicate_canonical_gene"


def test_rna_protein_source_api_is_exported():
    assert oncoref.rna_protein is rna_protein
    assert oncoref.rna_protein_calibration_sources is (rna_protein.rna_protein_calibration_sources)


def test_cptac_calibration_table_is_canonical_complete_and_self_consistent():
    frame = rna_protein.rna_protein_calibrations()

    assert tuple(frame.columns) == rna_protein.CPTAC_CALIBRATION_COLUMNS
    assert len(frame) == 115146
    assert frame["canonical_gene_id"].nunique() == 15087
    assert set(frame["cptac_cohort"]) == set(rna_protein.CPTAC_CALIBRATION_COHORTS)
    assert not frame.duplicated(["cptac_cohort", "canonical_gene_id"]).any()
    assert frame["canonical_gene_id"].str.fullmatch(r"ENSG[0-9]{11}").all()
    assert frame["canonical_proteoform_id"].notna().all()
    numeric = frame.select_dtypes(include="number").to_numpy()
    assert np.isfinite(numeric[~np.isnan(numeric)]).all()

    assert frame["proteoform_member_count"].ge(1).all()
    assert frame["proteoform_member_count"].gt(1).any()

    expected_detection = frame["n_protein_observed"] / frame["n_rna_observed"]
    assert np.allclose(frame["protein_detection_rate"], expected_detection)
    fitted_detection = frame.loc[frame["detection_model_status"].eq("fit")]
    assert (
        fitted_detection[
            [
                "detection_logit_intercept",
                "detection_logit_slope",
                "detection_auc",
                "detection_brier_score_in_sample",
            ]
        ]
        .notna()
        .all(axis=None)
    )
    thresholds = fitted_detection.dropna(subset=["rna_at_50pct_detection"])
    assert thresholds["detection_logit_slope"].gt(0).all()
    assert (
        thresholds["rna_at_50pct_detection"]
        .between(thresholds["rna_min"], thresholds["rna_max"])
        .all()
    )

    fitted_quantitative = frame.loc[frame["quantitative_model_status"].eq("fit")]
    required_quality = [
        "quantitative_intercept",
        "quantitative_slope",
        "quantitative_slope_standard_error",
        "pearson_r",
        "r_squared_in_sample",
        "rmse_in_sample",
    ]
    assert fitted_quantitative[required_quality].notna().all(axis=None)
    held_out = fitted_quantitative.dropna(subset=["rmse_leave_one_out"])
    assert (held_out["rmse_leave_one_out"] + 1e-9 >= held_out["rmse_in_sample"]).all()
    assert np.allclose(
        fitted_quantitative["r_squared_in_sample"],
        fitted_quantitative["pearson_r"] ** 2,
        atol=1e-8,
    )


def test_cptac_calibration_filters_resolve_gene_and_proteoform_ids():
    tp53 = rna_protein.rna_protein_calibrations(gene="TP53")

    assert len(tp53) == 10
    assert set(tp53["canonical_gene_id"]) == {"ENSG00000141510"}
    assert rna_protein.rna_protein_calibrations(
        cancer_code=" kirc ", cptac_cohort="ccrcc", gene="ENSG00000141510.17"
    )["cptac_cohort"].tolist() == ["CCRCC"]
    assert rna_protein.rna_protein_calibrations(gene="not-a-real-gene").empty

    grouped = rna_protein.rna_protein_calibrations().loc[
        lambda frame: frame["proteoform_member_count"].gt(1)
    ]
    example_proteoform = grouped["canonical_proteoform_id"].iloc[0]
    proteoform_rows = rna_protein.rna_protein_calibrations(proteoform=example_proteoform)
    assert not proteoform_rows.empty
    assert set(proteoform_rows["canonical_proteoform_id"]) == {example_proteoform}
    assert proteoform_rows["proteoform_member_count"].ge(2).all()


def test_cptac_sample_manifest_matches_model_denominators_and_sources():
    samples = rna_protein.rna_protein_calibration_samples()
    calibrations = rna_protein.rna_protein_calibrations()
    sources = rna_protein.rna_protein_calibration_sources()

    assert tuple(samples.columns) == rna_protein.CPTAC_CALIBRATION_SAMPLE_COLUMNS
    assert len(samples) == 1023
    assert not samples.duplicated(["cptac_cohort", "sample_id"]).any()
    sample_counts = samples.groupby("cptac_cohort")["sample_id"].size()
    model_counts = calibrations.groupby("cptac_cohort")["n_matched_samples"].first()
    pd.testing.assert_series_equal(sample_counts, model_counts, check_names=False)
    assert set(samples["rna_source_id"]) <= set(sources["source_id"])
    assert set(samples["protein_source_id"]) <= set(sources["source_id"])
    assert len(rna_protein.rna_protein_calibration_samples(cancer_code="KIRC")) == 103


def test_calibration_model_fits_keep_detection_and_abundance_separate():
    rna = np.arange(20, dtype=float)
    protein = np.where(rna >= 10, 2.0 + 3.0 * rna, np.nan)

    detection = fitter.fit_detection_model(rna, protein)
    quantitative = fitter.fit_quantitative_model(rna, protein)

    assert detection["detection_model_status"] == "fit"
    assert detection["detection_logit_slope"] > 0
    assert detection["detection_auc"] == pytest.approx(1.0)
    assert detection["rna_at_50pct_detection"] == pytest.approx(9.5, abs=1.0)
    assert quantitative["quantitative_model_status"] == "fit"
    assert quantitative["quantitative_intercept"] == pytest.approx(2.0)
    assert quantitative["quantitative_slope"] == pytest.approx(3.0)
    assert quantitative["pearson_r"] == pytest.approx(1.0)
    assert quantitative["rmse_leave_one_out"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "protein,expected",
    [
        (np.arange(20, dtype=float), "all_detected"),
        (np.full(20, np.nan), "all_missing"),
        (np.array([1.0] * 4 + [np.nan] * 16), "insufficient_events"),
    ],
)
def test_detection_model_reports_nonfittable_missingness_states(protein, expected):
    result = fitter.fit_detection_model(np.arange(20, dtype=float), protein)

    assert result["detection_model_status"] == expected
    assert pd.isna(result["detection_logit_intercept"])
    assert pd.isna(result["detection_logit_slope"])


def test_calibration_gzip_writer_is_byte_deterministic(tmp_path):
    frame = pd.DataFrame({"gene": ["B", "A"], "value": [1.25, np.nan]})
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"

    assert fitter.write_deterministic_gzip_csv(frame, first) == (
        fitter.write_deterministic_gzip_csv(frame, second)
    )
    assert first.read_bytes() == second.read_bytes()


def test_rna_protein_calibration_api_is_exported():
    assert oncoref.rna_protein_calibrations is rna_protein.rna_protein_calibrations
    assert oncoref.rna_protein_calibration_samples is (rna_protein.rna_protein_calibration_samples)


def test_hpa_prior_source_manifest_pins_archives_and_extracted_tables():
    sources = rna_protein.rna_protein_hpa_prior_sources()

    assert tuple(sources.columns) == rna_protein.HPA_PRIOR_SOURCE_COLUMNS
    assert len(sources) == 2
    assert set(sources["modality"]) == {"rna", "protein"}
    assert set(sources["source_version"]) == {"v23"}
    assert set(sources["source_class"]) == {rna_protein.HPA_PRIOR_SOURCE_CLASS}
    assert set(sources["license_id"]) == {"CC-BY-SA-3.0"}
    assert sources["archive_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert sources["extracted_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert rna_protein.rna_protein_hpa_prior_sources(modality=" RNA ")["source_id"].tolist() == [
        "hpa-v23-rna-tissue-consensus"
    ]


def test_hpa_prior_table_is_canonical_versioned_and_self_consistent():
    priors = rna_protein.rna_protein_hpa_priors()
    sources = rna_protein.rna_protein_hpa_prior_sources()

    assert tuple(priors.columns) == rna_protein.HPA_PRIOR_COLUMNS
    assert len(priors) == 13461
    assert priors["canonical_gene_id"].nunique() == 13461
    assert set(priors["prior_version"]) == {rna_protein.HPA_PRIOR_VERSION}
    assert set(priors["source_class"]) == {rna_protein.HPA_PRIOR_SOURCE_CLASS}
    assert set(priors["ihc_reliability"]) == {
        "Approved",
        "Enhanced",
        "Supported",
        "Uncertain",
    }
    assert priors["canonical_gene_id"].str.fullmatch(r"ENSG[0-9]{11}").all()
    assert priors["canonical_proteoform_id"].notna().all()
    assert priors["proteoform_member_count"].ge(1).all()
    assert priors["proteoform_member_count"].gt(1).any()
    assert set(priors["rna_source_id"]) | set(priors["protein_source_id"]) <= set(
        sources["source_id"]
    )

    level_sum = priors[
        ["n_not_detected_tissues", "n_low_tissues", "n_medium_tissues", "n_high_tissues"]
    ].sum(axis=1)
    assert level_sum.eq(priors["n_matched_tissues"]).all()
    assert (
        priors["n_detected_tissues"]
        .eq(priors["n_low_tissues"] + priors["n_medium_tissues"] + priors["n_high_tissues"])
        .all()
    )
    assert np.allclose(
        priors["ihc_detected_tissue_rate"],
        priors["n_detected_tissues"] / priors["n_matched_tissues"],
    )
    assert priors["n_nonordinal_ihc_observations_excluded"].sum() == 245

    numeric = priors.select_dtypes(include="number").to_numpy()
    assert np.isfinite(numeric[~np.isnan(numeric)]).all()
    fitted = priors.loc[priors["detection_prior_status"].eq("fit")]
    assert len(fitted) == 5667
    assert (
        fitted[
            [
                "detection_logit_intercept",
                "detection_logit_slope",
                "detection_auc",
                "detection_brier_score_in_sample",
            ]
        ]
        .notna()
        .all(axis=None)
    )
    thresholds = fitted.dropna(subset=["rna_at_50pct_ihc_detection"])
    assert thresholds["detection_logit_slope"].gt(0).all()
    assert (
        thresholds["rna_at_50pct_ihc_detection"]
        .between(thresholds["rna_min"], thresholds["rna_max"])
        .all()
    )
    associations = priors["rna_ihc_spearman_rho"].dropna()
    assert associations.between(-1.0, 1.0).all()


def test_hpa_prior_filters_resolve_gene_proteoform_reliability_and_status():
    tp53 = rna_protein.rna_protein_hpa_priors(gene="TP53")

    assert tp53["canonical_gene_id"].tolist() == ["ENSG00000141510"]
    assert tp53["ihc_reliability"].tolist() == ["Enhanced"]
    assert rna_protein.rna_protein_hpa_priors(gene="not-a-real-gene").empty
    assert rna_protein.rna_protein_hpa_priors(gene="MATR3").empty
    assert not rna_protein.rna_protein_hpa_priors(
        ihc_reliability=" approved ", detection_status="FIT"
    ).empty

    grouped = rna_protein.rna_protein_hpa_priors().loc[
        lambda frame: frame["proteoform_member_count"].gt(1)
    ]
    example = grouped["canonical_proteoform_id"].iloc[0]
    selected = rna_protein.rna_protein_hpa_priors(proteoform=example)
    assert not selected.empty
    assert set(selected["canonical_proteoform_id"]) == {example}


def test_hpa_builder_uses_exact_tissue_labels_and_excludes_nonordinal_levels():
    tissues = [f"tissue-{index:02d}" for index in range(43)]
    rna = pd.DataFrame(
        {
            "Gene": ["ENSG00000141510"] * 44,
            "Gene name": ["TP53"] * 44,
            "Tissue": [*tissues, "stomach"],
            "nTPM": np.arange(44, dtype=float),
        }
    )
    ihc = pd.DataFrame(
        {
            "Gene": ["ENSG00000141510"] * 45,
            "Gene name": ["TP53"] * 45,
            "Tissue": [*tissues, tissues[0], "stomach 1"],
            "Cell type": ["representative"] * 43 + ["gradient", "representative"],
            "Level": ["Not detected"] * 20 + ["Medium"] * 23 + ["Ascending", "High"],
            "Reliability": ["Enhanced"] * 45,
        }
    )

    pairs, audit, collisions = hpa_builder.canonical_hpa_tissue_pairs(rna, ihc)

    assert len(pairs) == 43
    assert set(pairs["Tissue"]) == set(tissues)
    assert "stomach" not in set(pairs["Tissue"])
    assert collisions == 0
    assert audit["n_ihc_cell_type_observations"].tolist() == [44]
    assert audit["n_nonordinal_ihc_observations_excluded"].tolist() == [1]


def test_hpa_builder_rejects_duplicate_source_observations():
    rna = pd.DataFrame(
        {
            "Gene": ["ENSG00000141510", "ENSG00000141510"],
            "Gene name": ["TP53", "TP53"],
            "Tissue": ["liver", "liver"],
            "nTPM": [1.0, 2.0],
        }
    )
    ihc = pd.DataFrame(
        {
            "Gene": ["ENSG00000141510"],
            "Gene name": ["TP53"],
            "Tissue": ["liver"],
            "Cell type": ["hepatocytes"],
            "Level": ["High"],
            "Reliability": ["Enhanced"],
        }
    )

    with pytest.raises(ValueError, match="unique by source gene and tissue"):
        hpa_builder.canonical_hpa_tissue_pairs(rna, ihc)


def test_hpa_detection_prior_fits_only_valid_tissue_pairs():
    rna = np.arange(21, dtype=float)
    ordinal = np.array([0.0] * 10 + [2.0] * 10 + [np.nan])

    result = hpa_builder.fit_ihc_detection_prior(rna, ordinal)

    assert result["detection_prior_status"] == "fit"
    assert result["detection_logit_slope"] > 0
    assert result["detection_auc"] == pytest.approx(1.0)
    assert result["rna_ihc_spearman_rho"] > 0


def test_hpa_archive_loader_rejects_unpinned_bytes(tmp_path):
    (tmp_path / "rna_tissue_consensus.tsv.zip").write_bytes(b"not the pinned archive")
    (tmp_path / "normal_tissue.tsv.zip").write_bytes(b"not the pinned archive")

    with pytest.raises(ValueError, match="archive byte size"):
        hpa_builder.load_pinned_hpa_archives(tmp_path)


def test_hpa_prior_gzip_writer_is_byte_deterministic(tmp_path):
    frame = pd.DataFrame({"gene": ["B", "A"], "value": [1.25, np.nan]})
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"

    assert hpa_builder.write_deterministic_gzip_csv(frame, first) == (
        hpa_builder.write_deterministic_gzip_csv(frame, second)
    )
    assert first.read_bytes() == second.read_bytes()


def test_hpa_prior_api_is_exported():
    assert oncoref.rna_protein_hpa_prior_sources is rna_protein.rna_protein_hpa_prior_sources
    assert oncoref.rna_protein_hpa_priors is rna_protein.rna_protein_hpa_priors
