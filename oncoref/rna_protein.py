# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Version-pinned RNA-to-protein calibration sources.

The CPTAC rows describe matched tumor measurements that can support quantitative
calibration.  They are intentionally distinct from the HPA tissue-level prior,
which is added as a separately typed evidence surface rather than treated as
patient-matched data.
"""

from __future__ import annotations

import re

import pandas as pd

from .cancer_types import cancer_type_codes
from .gene_ids import canonical_gene_id
from .load_dataset import get_data

RNA_PROTEIN_SOURCE_COLUMNS = (
    "source_id",
    "cptac_cohort",
    "cancer_code",
    "modality",
    "source_class",
    "source_scale",
    "source_record",
    "source_file",
    "byte_size",
    "checksum",
    "source_url",
    "license_id",
    "processing_software",
    "processing_version",
    "processing_commit",
    "tissue_scope",
)

CPTAC_CALIBRATION_COHORTS = (
    "BRCA",
    "CCRCC",
    "COAD",
    "GBM",
    "HNSCC",
    "LSCC",
    "LUAD",
    "OV",
    "PDAC",
    "UCEC",
)

RNA_PROTEIN_SOURCE_CLASSES = ("cptac_matched_quantitative",)
RNA_PROTEIN_SOURCE_SCALES = {
    "rna": "log2_uq_rsem_plus_1",
    "protein": "log2_tmt_reference_intensity_normalized",
}

CPTAC_CALIBRATION_VERSION = "cptac-bcm-cohort-v1"
CPTAC_CALIBRATION_SAMPLE_COLUMNS = (
    "calibration_version",
    "source_class",
    "cptac_cohort",
    "cancer_code",
    "sample_id",
    "rna_source_sample_id",
    "protein_source_sample_id",
    "rna_source_id",
    "protein_source_id",
    "source_record",
    "model_role",
)
CPTAC_CALIBRATION_COLUMNS = (
    "calibration_version",
    "source_class",
    "cptac_cohort",
    "cancer_code",
    "canonical_gene_id",
    "canonical_proteoform_id",
    "proteoform_member_count",
    "source_gene_id",
    "rna_scale",
    "protein_scale",
    "n_matched_samples",
    "n_rna_observed",
    "n_protein_observed",
    "protein_detection_rate",
    "rna_min",
    "rna_max",
    "protein_min",
    "protein_max",
    "detection_model_status",
    "detection_logit_intercept",
    "detection_logit_slope",
    "rna_at_50pct_detection",
    "detection_auc",
    "detection_brier_score_in_sample",
    "quantitative_model_status",
    "quantitative_intercept",
    "quantitative_slope",
    "quantitative_slope_standard_error",
    "pearson_r",
    "r_squared_in_sample",
    "rmse_in_sample",
    "rmse_leave_one_out",
    "rna_source_id",
    "protein_source_id",
    "source_record",
    "model_scope",
)

_DATASET = "rna-protein-calibration-sources"
_CALIBRATION_DATASET = "rna-protein-calibrations"
_SAMPLE_DATASET = "rna-protein-calibration-samples"
_MD5 = re.compile(r"md5:[0-9a-f]{32}\Z")


def _validate_rna_protein_calibration_sources(frame: pd.DataFrame) -> None:
    """Validate the source manifest used to build matched CPTAC calibrations."""

    if tuple(frame.columns) != RNA_PROTEIN_SOURCE_COLUMNS:
        raise ValueError(f"{_DATASET}.csv columns must be exactly {RNA_PROTEIN_SOURCE_COLUMNS}")
    if frame.empty:
        raise ValueError("RNA-to-protein source manifest must not be empty")

    for column in ("source_id", "source_file"):
        values = frame[column].astype(str).str.strip()
        if values.eq("").any() or values.duplicated().any():
            raise ValueError(f"{column} values must be nonempty and unique")

    if set(frame["cptac_cohort"]) != set(CPTAC_CALIBRATION_COHORTS):
        raise ValueError("CPTAC calibration cohort coverage is incomplete")
    grouped = frame.groupby("cptac_cohort", sort=False)["modality"].agg(list)
    invalid_pairs = {
        cohort: sorted(modalities)
        for cohort, modalities in grouped.items()
        if sorted(modalities) != ["protein", "rna"]
    }
    if invalid_pairs:
        raise ValueError(
            f"each CPTAC cohort requires one RNA and one protein source: {invalid_pairs}"
        )

    invalid_classes = set(frame["source_class"]) - set(RNA_PROTEIN_SOURCE_CLASSES)
    if invalid_classes:
        raise ValueError(f"invalid RNA-to-protein source classes: {sorted(invalid_classes)}")
    invalid_modalities = set(frame["modality"]) - set(RNA_PROTEIN_SOURCE_SCALES)
    if invalid_modalities:
        raise ValueError(f"invalid RNA-to-protein modalities: {sorted(invalid_modalities)}")
    wrong_scale = frame.apply(
        lambda row: RNA_PROTEIN_SOURCE_SCALES.get(row["modality"]) != row["source_scale"],
        axis=1,
    )
    if wrong_scale.any():
        raise ValueError("source_scale must match the declared RNA or protein measurement modality")

    unknown_codes = set(frame["cancer_code"]) - set(cancer_type_codes())
    if unknown_codes:
        raise ValueError(f"unknown cancer_code values: {sorted(unknown_codes)}")
    if not frame["checksum"].astype(str).map(_MD5.fullmatch).map(bool).all():
        raise ValueError("source checksums must be lowercase md5:<32 hex> values")
    sizes = pd.to_numeric(frame["byte_size"], errors="coerce")
    if sizes.isna().any() or sizes.le(0).any() or sizes.mod(1).ne(0).any():
        raise ValueError("source byte_size values must be positive integers")
    if not frame["source_url"].astype(str).str.startswith("https://").all():
        raise ValueError("source_url values must use HTTPS")
    if set(frame["source_record"]) != {"10.5281/zenodo.8394329"}:
        raise ValueError("CPTAC calibration sources must use the pinned Zenodo record")
    if set(frame["processing_version"]) != {"1.5.14"}:
        raise ValueError("CPTAC processing version must stay pinned")
    commits = frame["processing_commit"].astype(str)
    if not commits.str.fullmatch(r"[0-9a-f]{40}").all() or commits.nunique() != 1:
        raise ValueError("CPTAC processing commit must be one pinned full Git SHA")
    if set(frame["tissue_scope"]) != {"tumor"}:
        raise ValueError("the matched quantitative calibration sources must be tumor-only")


def rna_protein_calibration_sources(
    *, cptac_cohort: str | None = None, modality: str | None = None
) -> pd.DataFrame:
    """Return checksum-pinned source files for matched CPTAC calibration.

    ``cptac_cohort`` and ``modality`` are exact, case-insensitive filters.  The
    returned rows pin both the immutable Zenodo files and the CPTAC Python code
    version used to interpret their sample and measurement conventions.
    """

    frame = get_data(_DATASET)
    _validate_rna_protein_calibration_sources(frame)
    if cptac_cohort is not None:
        wanted = str(cptac_cohort).strip().casefold()
        frame = frame.loc[frame["cptac_cohort"].astype(str).str.casefold().eq(wanted)]
    if modality is not None:
        wanted = str(modality).strip().casefold()
        frame = frame.loc[frame["modality"].astype(str).str.casefold().eq(wanted)]
    return frame.reset_index(drop=True)


def _validate_cptac_calibrations(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != CPTAC_CALIBRATION_COLUMNS:
        raise ValueError(
            f"{_CALIBRATION_DATASET}.csv.gz columns must be exactly {CPTAC_CALIBRATION_COLUMNS}"
        )
    if frame.empty:
        raise ValueError("CPTAC RNA-to-protein calibration table must not be empty")
    if frame.duplicated(["cptac_cohort", "canonical_gene_id"]).any():
        raise ValueError("CPTAC calibration rows must be unique by cohort and canonical gene")
    if set(frame["calibration_version"]) != {CPTAC_CALIBRATION_VERSION}:
        raise ValueError("CPTAC calibration version does not match the reader")
    if set(frame["source_class"]) != {"cptac_matched_quantitative"}:
        raise ValueError("CPTAC calibration rows have an invalid source class")
    if set(frame["cptac_cohort"]) != set(CPTAC_CALIBRATION_COHORTS):
        raise ValueError("CPTAC calibration cohort coverage is incomplete")
    if not frame["canonical_gene_id"].astype(str).str.fullmatch(r"ENSG[0-9]{11}").all():
        raise ValueError("CPTAC calibration rows require canonical Ensembl gene IDs")
    if frame["canonical_proteoform_id"].isna().any():
        raise ValueError("CPTAC calibration rows require canonical proteoform IDs")
    if frame["proteoform_member_count"].lt(1).any():
        raise ValueError("CPTAC proteoform member counts must be positive")
    if not frame["protein_detection_rate"].between(0.0, 1.0).all():
        raise ValueError("protein_detection_rate must be in [0, 1]")
    if (frame["n_protein_observed"] > frame["n_rna_observed"]).any() or (
        frame["n_rna_observed"] > frame["n_matched_samples"]
    ).any():
        raise ValueError("CPTAC calibration observation counts are inconsistent")
    detection_statuses = {
        "fit",
        "all_detected",
        "all_missing",
        "insufficient_events",
        "constant_rna",
        "fit_failed",
    }
    quantitative_statuses = {
        "fit",
        "insufficient_pairs",
        "constant_rna",
        "constant_protein",
    }
    if set(frame["detection_model_status"]) - detection_statuses:
        raise ValueError("CPTAC calibration rows have an invalid detection model status")
    if set(frame["quantitative_model_status"]) - quantitative_statuses:
        raise ValueError("CPTAC calibration rows have an invalid quantitative model status")
    detection_fit = frame["detection_model_status"].eq("fit")
    if (
        frame.loc[
            detection_fit,
            ["detection_logit_intercept", "detection_logit_slope"],
        ]
        .isna()
        .any(axis=None)
    ):
        raise ValueError("fitted CPTAC detection rows require coefficients")
    quantitative_fit = frame["quantitative_model_status"].eq("fit")
    quantitative_fields = [
        "quantitative_intercept",
        "quantitative_slope",
        "quantitative_slope_standard_error",
        "pearson_r",
        "r_squared_in_sample",
        "rmse_in_sample",
    ]
    if frame.loc[quantitative_fit, quantitative_fields].isna().any(axis=None):
        raise ValueError("fitted CPTAC quantitative rows require coefficients and quality metrics")


def rna_protein_calibrations(
    *,
    cancer_code: str | None = None,
    cptac_cohort: str | None = None,
    gene: str | None = None,
    proteoform: str | None = None,
) -> pd.DataFrame:
    """Return cohort-specific matched CPTAC RNA/protein calibrations.

    Quantitative coefficients predict TMT reference-normalized log2 abundance
    from upper-quartile-normalized RSEM log2(x + 1) *within the named cohort*.
    Detection coefficients model whether the TMT measurement is observed; a
    missing protein value is never imputed or relabeled as biological absence.
    Filters are exact and case-insensitive. ``gene`` accepts any identifier
    supported by :func:`oncoref.gene_ids.canonical_gene_id`.
    """

    frame = get_data(_CALIBRATION_DATASET)
    _validate_cptac_calibrations(frame)
    for column, value in (("cancer_code", cancer_code), ("cptac_cohort", cptac_cohort)):
        if value is not None:
            wanted = str(value).strip().casefold()
            frame = frame.loc[frame[column].astype(str).str.casefold().eq(wanted)]
    if gene is not None:
        wanted_gene = canonical_gene_id(gene)
        if wanted_gene is None:
            return frame.iloc[0:0].reset_index(drop=True)
        frame = frame.loc[frame["canonical_gene_id"].eq(wanted_gene)]
    if proteoform is not None:
        wanted_proteoform = str(proteoform).strip().casefold()
        frame = frame.loc[
            frame["canonical_proteoform_id"].astype(str).str.casefold().eq(wanted_proteoform)
        ]
    return frame.reset_index(drop=True)


def rna_protein_calibration_samples(
    *, cancer_code: str | None = None, cptac_cohort: str | None = None
) -> pd.DataFrame:
    """Return the explicit matched-patient manifest behind CPTAC calibrations."""

    frame = get_data(_SAMPLE_DATASET)
    if tuple(frame.columns) != CPTAC_CALIBRATION_SAMPLE_COLUMNS:
        raise ValueError(
            f"{_SAMPLE_DATASET}.csv.gz columns must be exactly {CPTAC_CALIBRATION_SAMPLE_COLUMNS}"
        )
    if frame.empty or frame.duplicated(["cptac_cohort", "sample_id"]).any():
        raise ValueError("CPTAC calibration sample rows must be nonempty and unique")
    if set(frame["calibration_version"]) != {CPTAC_CALIBRATION_VERSION}:
        raise ValueError("CPTAC calibration sample version does not match the reader")
    if set(frame["source_class"]) != {"cptac_matched_quantitative"}:
        raise ValueError("CPTAC calibration samples have an invalid source class")
    if set(frame["cptac_cohort"]) != set(CPTAC_CALIBRATION_COHORTS):
        raise ValueError("CPTAC calibration sample cohort coverage is incomplete")
    if set(frame["model_role"]) != {"cohort_calibration_observation"}:
        raise ValueError("CPTAC calibration samples have an invalid model role")
    for column, value in (("cancer_code", cancer_code), ("cptac_cohort", cptac_cohort)):
        if value is not None:
            wanted = str(value).strip().casefold()
            frame = frame.loc[frame[column].astype(str).str.casefold().eq(wanted)]
    return frame.reset_index(drop=True)


__all__ = [
    "CPTAC_CALIBRATION_COHORTS",
    "CPTAC_CALIBRATION_COLUMNS",
    "CPTAC_CALIBRATION_SAMPLE_COLUMNS",
    "CPTAC_CALIBRATION_VERSION",
    "RNA_PROTEIN_SOURCE_CLASSES",
    "RNA_PROTEIN_SOURCE_COLUMNS",
    "RNA_PROTEIN_SOURCE_SCALES",
    "rna_protein_calibration_samples",
    "rna_protein_calibration_sources",
    "rna_protein_calibrations",
]
