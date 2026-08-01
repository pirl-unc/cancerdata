# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tumor-reference expression artifacts used by downstream classifiers.

Oncoref owns the versioned data and its interpretation. Trufflepig still owns
the decomposition algorithm that produced the TCGA artifact; importing these
tables does not create a runtime dependency on Trufflepig.

There are two distinct products:

* the TCGA table contains tumor-attributed TPM from per-sample decomposition;
* the historically named subtype table contains observed-TPM cohort summaries,
  not universal tumor-microenvironment deconvolution.

Use :func:`tumor_reference_expression_provenance` to inspect that distinction
without inferring a method from a filename.
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd

from .cancer_types import (
    cancer_type_codes,
    canonical_cohort_id,
    known_cohort_ids,
    resolve_cancer_type,
)
from .gene_qc import TECHNICAL_RNA_GROUPS, classify_gene_qc
from .load_dataset import get_data
from .version import DATA_VERSION

TUMOR_REFERENCE_VALUE_COLUMNS = (
    "tumor_tpm_median",
    "tumor_tpm_q1",
    "tumor_tpm_q3",
)
TUMOR_REFERENCE_SCALE_VALUES = ("classifier_tpm", "native")
TUMOR_REFERENCE_DERIVATION_METHODS = (
    "tme_deconvolution",
    "high_purity_passthrough",
    "observed_tpm_passthrough",
)
TUMOR_REFERENCE_DERIVATION_STATUSES = (
    "migrated_pinned_artifact",
    "legacy_artifact_source_inferred",
    "rebuilt_from_oncoref_source_matrix",
)
TUMOR_REFERENCE_SOURCE_SCALES = (
    "tumor_attributed_tpm",
    "observed_tpm",
    "clean_tpm_16_9_75",
)
TUMOR_REFERENCE_PROVENANCE_COLUMNS = (
    "artifact",
    "cancer_code",
    "subtype",
    "source_cohort",
    "derivation_method",
    "derivation_status",
    "source_scale",
    "processing_pipeline",
    "source_artifact",
    "source_artifact_sha256",
    "source_artifact_commit",
    "source_matrix_version",
    "output_artifact_sha256",
    "n_genes",
    "n_reference_samples",
    "notes",
)

_TCGA_DATASET = "tcga-deconvolved-expression"
_SUBTYPE_DATASET = "subtype-deconvolved-expression"
_PROVENANCE_DATASET = "tumor-reference-expression-provenance"
_DERIVATION_METHOD_BY_STATUS = {
    "migrated_pinned_artifact": "tme_deconvolution",
    "legacy_artifact_source_inferred": "observed_tpm_passthrough",
    "rebuilt_from_oncoref_source_matrix": "high_purity_passthrough",
}

_TCGA_COLUMNS = (
    "Ensembl_Gene_ID",
    "symbol",
    "cancer_code",
    *TUMOR_REFERENCE_VALUE_COLUMNS,
    "n_samples",
)
_SUBTYPE_COLUMNS = (
    "Ensembl_Gene_ID",
    "symbol",
    "cancer_code",
    "subtype",
    "source_cohort",
    *TUMOR_REFERENCE_VALUE_COLUMNS,
    "n_samples",
)

# These values are acquisition-cohort labels in the migrated artifact, not
# cancer ontology codes. Map them to the biological code explicitly.
_LEGACY_CANCER_CODES = {
    "BEATAML": "LAML",
    "NEC_UNSPEC": "NEC",
    "TARGET_AML": "LAML",
    "TARGET_NBL": "NBL",
    "TARGET_RT": "RT",
    "TARGET_WT": "WILMS",
}
_LEGACY_SUBTYPE_CODES = {
    "BEATAML_APL": "LAML_APL",
    "BEATAML_ELN_ADVERSE": "LAML_ELNadv",
    "BEATAML_ELN_FAVORABLE": "LAML_ELNfav",
    "BEATAML_ELN_INTERMEDIATE": "LAML_ELNint",
}
_LEGACY_SOURCE_COHORTS = {
    "TCGA_BRCA_PAM50": "TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50",
    "TCGA_HNSC_HPV": "TREEHOUSE_POLYA_25_01_TCGA_HNSC_HPV",
    "TCGA_LUAD_MUT": "TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT",
}


class TumorReferenceDataError(ValueError):
    """A tumor-reference artifact violates its published schema or value contract."""


def _canonical_code(value: object, *, subtype: bool = False) -> object:
    if pd.isna(value):
        return pd.NA
    raw = str(value).strip()
    if not raw:
        return pd.NA
    aliases = _LEGACY_SUBTYPE_CODES if subtype else _LEGACY_CANCER_CODES
    mapped = aliases.get(raw.upper(), raw)
    return resolve_cancer_type(mapped, strict=False) or mapped


def _canonical_source_cohort(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    raw = str(value).strip()
    if not raw:
        return pd.NA
    return canonical_cohort_id(_LEGACY_SOURCE_COHORTS.get(raw, raw))


def _canonicalize_codes(frame: pd.DataFrame) -> pd.DataFrame:
    frame["cancer_code"] = frame["cancer_code"].map(_canonical_code).astype("string")
    if "subtype" in frame.columns:
        frame["subtype"] = (
            frame["subtype"]
            .map(lambda value: _canonical_code(value, subtype=True))
            .astype("string")
        )
    if "source_cohort" in frame.columns:
        frame["source_cohort"] = (
            frame["source_cohort"].map(_canonical_source_cohort).astype("string")
        )
    return frame


def _with_stable_schema(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out[list(columns)]


def _logical_gene_key(frame: pd.DataFrame) -> pd.Series:
    symbols = frame["symbol"].astype("string").str.strip().str.upper()
    fallback = "symbol:" + symbols
    if "Ensembl_Gene_ID" not in frame.columns:
        return fallback
    gene_ids = frame["Ensembl_Gene_ID"].astype("string").str.strip()
    has_gene_id = gene_ids.notna() & gene_ids.ne("")
    return ("id:" + gene_ids).where(has_gene_id, fallback)


def _validate_identities(frame: pd.DataFrame, *, dataset: str) -> None:
    known_cancers = set(cancer_type_codes())
    values = set(frame["cancer_code"].dropna().astype(str))
    unknown = sorted(values - known_cancers)
    if unknown:
        raise TumorReferenceDataError(f"{dataset} contains unknown cancer_code values: {unknown}")

    if "source_cohort" in frame.columns:
        values = set(frame["source_cohort"].dropna().astype(str))
        unknown = sorted(values - set(known_cohort_ids()))
        if unknown:
            raise TumorReferenceDataError(
                f"{dataset} contains unknown source_cohort values: {unknown}"
            )


def _validate_reference(
    frame: pd.DataFrame,
    *,
    dataset: str,
    identity_columns: tuple[str, ...],
) -> pd.DataFrame:
    required = {
        "symbol",
        "cancer_code",
        "n_samples",
        *TUMOR_REFERENCE_VALUE_COLUMNS,
        *identity_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TumorReferenceDataError(f"{dataset} lacks required columns: {missing}")

    if frame["symbol"].isna().any() or frame["symbol"].astype(str).str.strip().eq("").any():
        raise TumorReferenceDataError(f"{dataset} contains missing gene symbols")
    required_identities = ["cancer_code", *[c for c in identity_columns if c != "subtype"]]
    for column in required_identities:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise TumorReferenceDataError(f"{dataset} contains missing {column} values")
    _validate_identities(frame, dataset=dataset)

    out = frame
    numeric_columns = [*TUMOR_REFERENCE_VALUE_COLUMNS, "n_samples"]
    numeric = out[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise TumorReferenceDataError(f"{dataset} contains missing or non-finite numeric values")
    if (numeric[list(TUMOR_REFERENCE_VALUE_COLUMNS)] < 0).any().any():
        count = int((numeric[list(TUMOR_REFERENCE_VALUE_COLUMNS)] < 0).sum().sum())
        raise TumorReferenceDataError(f"{dataset} contains {count} negative TPM values")
    invalid_quantiles = (numeric["tumor_tpm_q1"] > numeric["tumor_tpm_median"]) | (
        numeric["tumor_tpm_median"] > numeric["tumor_tpm_q3"]
    )
    if invalid_quantiles.any():
        raise TumorReferenceDataError(
            f"{dataset} contains {int(invalid_quantiles.sum())} rows with Q1 > median or median > Q3"
        )
    samples = numeric["n_samples"]
    if ((samples <= 0) | (samples % 1 != 0)).any():
        raise TumorReferenceDataError(f"{dataset} contains invalid sample counts")

    out[numeric_columns] = numeric
    logical_keys = out[list(identity_columns)].copy()
    logical_keys.insert(0, "_gene", _logical_gene_key(out))
    duplicate_count = int(logical_keys.duplicated().sum())
    if duplicate_count:
        raise TumorReferenceDataError(
            f"{dataset} contains {duplicate_count} duplicate logical gene rows"
        )
    return out


def _classifier_tpm(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    out = frame
    symbols = out["symbol"].fillna("").astype(str).str.strip()
    if "Ensembl_Gene_ID" in out.columns:
        gene_ids = out["Ensembl_Gene_ID"].fillna("").astype(str).str.split(".").str[0].str.strip()
        groups = (
            classify_gene_qc(symbol, ensembl_id=gene_id).group
            for symbol, gene_id in zip(symbols, gene_ids)
        )
    else:
        groups = (classify_gene_qc(symbol).group for symbol in symbols)
    technical = pd.Series(
        (group in TECHNICAL_RNA_GROUPS for group in groups),
        index=out.index,
        dtype=bool,
    )
    out.loc[technical, list(TUMOR_REFERENCE_VALUE_COLUMNS)] = 0.0

    groupers = [out[column] for column in group_columns]
    medians = out["tumor_tpm_median"]
    totals = medians.groupby(groupers, dropna=False).transform("sum")
    if (totals <= 0).any():
        raise TumorReferenceDataError("tumor-reference group has no positive median TPM mass")
    scales = 1_000_000.0 / totals
    for column in TUMOR_REFERENCE_VALUE_COLUMNS:
        out[column] *= scales
    return out


def _validate_scale(scale: str) -> Literal["classifier_tpm", "native"]:
    if scale not in TUMOR_REFERENCE_SCALE_VALUES:
        raise ValueError(f"scale must be one of {TUMOR_REFERENCE_SCALE_VALUES}, got {scale!r}")
    return scale


def _filter_rows(
    frame: pd.DataFrame,
    *,
    cancer_code: str | None,
    subtype_code: str | None = None,
    source_cohort: str | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    if cancer_code is not None:
        requested = str(_canonical_code(cancer_code))
        mask &= frame["cancer_code"].astype(str).eq(requested)
    if subtype_code is not None:
        requested = str(_canonical_code(subtype_code, subtype=True))
        mask &= frame["subtype"].astype(str).eq(requested)
    if source_cohort is not None:
        requested = str(_canonical_source_cohort(source_cohort))
        mask &= frame["source_cohort"].astype(str).eq(requested)
    return frame.loc[mask].reset_index(drop=True)


def tcga_deconvolved_expression(
    cancer_code: str | None = None,
    *,
    scale: Literal["classifier_tpm", "native"] = "classifier_tpm",
) -> pd.DataFrame:
    """Return the per-gene TCGA tumor-attributed reference table.

    ``scale="classifier_tpm"`` removes technical-RNA distortion within each
    cancer code and scales its median column to one million; Q1 and Q3 receive
    the same group scale. ``scale="native"`` preserves the migrated numeric
    values. Both modes canonicalize cancer codes and validate the artifact.
    """
    scale = _validate_scale(scale)
    frame = _with_stable_schema(get_data(_TCGA_DATASET, copy=False), _TCGA_COLUMNS)
    frame = _canonicalize_codes(frame)
    frame = _validate_reference(
        frame,
        dataset=_TCGA_DATASET,
        identity_columns=("cancer_code",),
    )
    frame = _filter_rows(frame, cancer_code=cancer_code)
    if scale == "classifier_tpm" and not frame.empty:
        frame = _classifier_tpm(frame, group_columns=("cancer_code",))
    frame.attrs["oncoref"] = {
        "dataset": _TCGA_DATASET,
        "data_version": DATA_VERSION,
        "scale": scale,
        "derivation_method": "tme_deconvolution",
    }
    return frame


def subtype_tumor_reference_expression(
    cancer_code: str | None = None,
    *,
    subtype_code: str | None = None,
    source_cohort: str | None = None,
    scale: Literal["classifier_tpm", "native"] = "classifier_tpm",
) -> pd.DataFrame:
    """Return subtype/cohort tumor-reference summaries.

    The source artifact's old name says "deconvolved", but its rows are
    observed-TPM passthrough summaries. Physical sources remain separate during
    filtering and normalization. See :func:`tumor_reference_expression_provenance`
    for the derivation recorded for each source.
    """
    scale = _validate_scale(scale)
    frame = _with_stable_schema(get_data(_SUBTYPE_DATASET, copy=False), _SUBTYPE_COLUMNS)
    frame = _canonicalize_codes(frame)
    frame = _validate_reference(
        frame,
        dataset=_SUBTYPE_DATASET,
        identity_columns=("cancer_code", "subtype", "source_cohort"),
    )
    frame = _filter_rows(
        frame,
        cancer_code=cancer_code,
        subtype_code=subtype_code,
        source_cohort=source_cohort,
    )
    if scale == "classifier_tpm" and not frame.empty:
        frame = _classifier_tpm(
            frame,
            group_columns=("cancer_code", "subtype", "source_cohort"),
        )
    frame.attrs["oncoref"] = {
        "dataset": _SUBTYPE_DATASET,
        "data_version": DATA_VERSION,
        "scale": scale,
        "derivation_method": "mixed_passthrough; inspect provenance",
    }
    return frame


def subtype_deconvolved_expression(*args, **kwargs) -> pd.DataFrame:
    """Compatibility name for :func:`subtype_tumor_reference_expression`.

    The historical filename is retained for downstream migration, but callers
    should use the preferred name because these rows are not uniformly
    tumor-microenvironment-deconvolved.
    """
    return subtype_tumor_reference_expression(*args, **kwargs)


def tumor_reference_expression_provenance(
    *,
    artifact: str | None = None,
    cancer_code: str | None = None,
    source_cohort: str | None = None,
) -> pd.DataFrame:
    """Return one derivation record per tumor-reference source/group."""
    frame = get_data(_PROVENANCE_DATASET, copy=False).copy()
    missing = sorted(set(TUMOR_REFERENCE_PROVENANCE_COLUMNS) - set(frame.columns))
    if missing:
        raise TumorReferenceDataError(f"{_PROVENANCE_DATASET} lacks required columns: {missing}")
    unknown = sorted(
        set(frame["derivation_method"].dropna().astype(str))
        - set(TUMOR_REFERENCE_DERIVATION_METHODS)
    )
    if unknown:
        raise TumorReferenceDataError(f"unknown tumor-reference derivation methods: {unknown}")
    unknown = sorted(
        set(frame["derivation_status"].dropna().astype(str))
        - set(TUMOR_REFERENCE_DERIVATION_STATUSES)
    )
    if unknown:
        raise TumorReferenceDataError(f"unknown tumor-reference derivation statuses: {unknown}")
    unknown = sorted(
        set(frame["source_scale"].dropna().astype(str)) - set(TUMOR_REFERENCE_SOURCE_SCALES)
    )
    if unknown:
        raise TumorReferenceDataError(f"unknown tumor-reference source scales: {unknown}")
    expected_methods = frame["derivation_status"].map(_DERIVATION_METHOD_BY_STATUS)
    if not frame["derivation_method"].astype(str).eq(expected_methods.astype(str)).all():
        raise TumorReferenceDataError(
            f"{_PROVENANCE_DATASET} contains inconsistent derivation method/status pairs"
        )
    required_text = (
        "artifact",
        "cancer_code",
        "source_cohort",
        "derivation_method",
        "derivation_status",
        "source_scale",
        "processing_pipeline",
        "source_artifact",
    )
    for column in required_text:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise TumorReferenceDataError(f"{_PROVENANCE_DATASET} contains missing {column}")
    for column in ("source_artifact_sha256", "output_artifact_sha256"):
        valid = frame[column].astype(str).map(lambda value: re.fullmatch(r"[0-9a-f]{64}", value))
        if valid.isna().any():
            raise TumorReferenceDataError(f"{_PROVENANCE_DATASET} contains invalid {column}")
    rebuilt = frame["derivation_status"].eq("rebuilt_from_oncoref_source_matrix")
    if frame.loc[rebuilt, "source_matrix_version"].isna().any():
        raise TumorReferenceDataError(
            f"{_PROVENANCE_DATASET} lacks source_matrix_version for rebuilt records"
        )
    frame = _canonicalize_codes(frame)
    _validate_identities(frame, dataset=_PROVENANCE_DATASET)
    duplicate_columns = ["artifact", "cancer_code", "subtype", "source_cohort"]
    if frame.duplicated(duplicate_columns).any():
        raise TumorReferenceDataError(f"{_PROVENANCE_DATASET} contains duplicate source records")
    for column in ("n_genes", "n_reference_samples"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values <= 0).any() or (values % 1 != 0).any():
            raise TumorReferenceDataError(f"{_PROVENANCE_DATASET} contains invalid {column}")
        frame[column] = values
    if artifact is not None:
        requested = str(artifact).removesuffix(".gz").removesuffix(".csv")
        frame = frame[frame["artifact"].astype(str).eq(requested)]
    frame = _filter_rows(
        frame,
        cancer_code=cancer_code,
        source_cohort=source_cohort,
    )
    frame.attrs["oncoref"] = {
        "dataset": _PROVENANCE_DATASET,
        "data_version": DATA_VERSION,
    }
    return frame[list(TUMOR_REFERENCE_PROVENANCE_COLUMNS)]
