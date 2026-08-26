# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Source-anchored clinical benefit and toxicity reference facts.

This module owns empirical evidence rows, not therapy selection.  Target-to-drug
registries and treatment panels remain downstream in :mod:`pirlygenes`; callers
must not infer clinical benefit or toxicity from target expression alone.
"""

from __future__ import annotations

import pandas as pd

from .load_dataset import get_data

THERAPY_BENEFIT_TIERS = (
    "curative",
    "durable_rfs",
    "major_survival",
    "high_response",
    "meaningful_pfs",
    "incremental",
    "modest",
    "unclear",
)

THERAPY_TOXICITY_TIERS = (
    "minimal",
    "low",
    "moderate",
    "high",
    "very_high",
    "unclear",
)

THERAPY_EVIDENCE_TRANSFER_VALUES = (
    "disease_matched",
    "cross_indication",
    "safety_signal_only",
)

THERAPY_EVIDENCE_COLUMNS = (
    "evidence_id",
    "agent",
    "agent_class",
    "target_symbol",
    "cancer_code",
    "subtype",
    "line_of_therapy",
    "setting",
    "endpoint_type",
    "endpoint_value",
    "benefit_tier",
    "toxicity_tier",
    "grade3_plus_ae_rate",
    "discontinuation_rate",
    "boxed_warning",
    "major_toxicities",
    "source_type",
    "source_id",
    "source_anchor",
    "source_url",
    "source_token",
    "evidence_transfer",
    "evidence_notes",
)

_DATASET = "therapy-benefit-toxicity-evidence"


def _nonempty_strings(series: pd.Series) -> set[str]:
    return set(series.dropna().astype(str).str.strip()) - {""}


def _filter_exact(frame: pd.DataFrame, column: str, value: str | None) -> pd.DataFrame:
    if value is None:
        return frame
    wanted = str(value).strip().casefold()
    values = frame[column].fillna("").astype(str).str.strip().str.casefold()
    return frame.loc[values.eq(wanted)]


def _validate_therapy_evidence(frame: pd.DataFrame) -> None:
    missing = set(THERAPY_EVIDENCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{_DATASET}.csv is missing columns: {sorted(missing)}")

    duplicate_ids = frame.loc[frame["evidence_id"].duplicated(), "evidence_id"].astype(str)
    if not duplicate_ids.empty:
        raise ValueError(f"duplicate therapy evidence_id values: {sorted(duplicate_ids)}")
    empty_ids = frame["evidence_id"].isna() | frame["evidence_id"].astype(str).str.strip().eq("")
    if empty_ids.any():
        raise ValueError("therapy evidence_id values must be nonempty")

    enumerations = {
        "benefit_tier": THERAPY_BENEFIT_TIERS,
        "toxicity_tier": THERAPY_TOXICITY_TIERS,
        "evidence_transfer": THERAPY_EVIDENCE_TRANSFER_VALUES,
    }
    for column, allowed in enumerations.items():
        invalid = _nonempty_strings(frame[column]) - set(allowed)
        if invalid:
            raise ValueError(f"invalid {column} values: {sorted(invalid)}")

    for column in ("source_token", "source_anchor", "source_url"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"therapy evidence {column} values must be nonempty")
    urls = frame["source_url"].astype(str).str.strip()
    if not urls.str.startswith(("https://", "http://")).all():
        raise ValueError("therapy evidence source_url values must use http(s)")

    postmarket = (
        frame["source_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("postmarket_signal")
    )
    for column in ("grade3_plus_ae_rate", "discontinuation_rate"):
        populated = frame[column].notna() & frame[column].astype(str).str.strip().ne("")
        if (postmarket & populated).any():
            bad = sorted(frame.loc[postmarket & populated, "evidence_id"].astype(str))
            raise ValueError(
                f"postmarket_signal rows must not provide incidence-like {column} values: {bad}"
            )


def therapy_benefit_toxicity_evidence(
    *,
    agent: str | None = None,
    cancer_code: str | None = None,
    subtype: str | None = None,
    line_of_therapy: str | None = None,
    source_type: str | None = None,
    include_transferred: bool = True,
) -> pd.DataFrame:
    """Curated clinical benefit/toxicity evidence with source provenance.

    Text filters are exact and case-insensitive.  When both ``cancer_code`` and
    ``subtype`` are supplied, disease-level rows with a blank subtype remain
    eligible alongside the exact subtype.  A subtype-only query returns exact
    subtype rows.  Set ``include_transferred=False`` to exclude cross-indication
    evidence that requires a separate eligibility check.
    """

    frame = get_data(_DATASET)
    _validate_therapy_evidence(frame)

    frame = _filter_exact(frame, "agent", agent)
    frame = _filter_exact(frame, "cancer_code", cancer_code)
    frame = _filter_exact(frame, "line_of_therapy", line_of_therapy)
    frame = _filter_exact(frame, "source_type", source_type)

    if subtype is not None:
        wanted = str(subtype).strip().casefold()
        values = frame["subtype"].fillna("").astype(str).str.strip().str.casefold()
        frame = frame.loc[values.eq(wanted) if cancer_code is None else values.isin(("", wanted))]

    if not include_transferred:
        transfer = frame["evidence_transfer"].fillna("").astype(str).str.strip().str.casefold()
        frame = frame.loc[transfer.ne("cross_indication")]

    return frame.loc[:, THERAPY_EVIDENCE_COLUMNS].copy().reset_index(drop=True)


__all__ = [
    "THERAPY_BENEFIT_TIERS",
    "THERAPY_EVIDENCE_COLUMNS",
    "THERAPY_EVIDENCE_TRANSFER_VALUES",
    "THERAPY_TOXICITY_TIERS",
    "therapy_benefit_toxicity_evidence",
]
