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

_DATASET = "rna-protein-calibration-sources"
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


__all__ = [
    "CPTAC_CALIBRATION_COHORTS",
    "RNA_PROTEIN_SOURCE_CLASSES",
    "RNA_PROTEIN_SOURCE_COLUMNS",
    "RNA_PROTEIN_SOURCE_SCALES",
    "rna_protein_calibration_sources",
]
