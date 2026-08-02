# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Structured entity-level driver spectra.

Diagnosis and molecular status are separate axes. This table records driver
events observed across a cancer entity; it never assigns an event to an
individual sample. Sample-level evidence lives in :mod:`oncoref.samples`.
"""

from __future__ import annotations

import pandas as pd

from .cancer_types import resolve_cancer_type
from .load_dataset import get_data


def cancer_driver_spectrum_df() -> pd.DataFrame:
    """All entity-driver relationships as a defensive dataframe copy.

    Each row is one observed event or explicitly unresolved group. Counts are
    study-level evidence and may overlap when two drivers co-occur in a case.
    """
    return get_data("cancer-entity-driver-spectrum").copy()


def cancer_driver_spectrum(cancer_type: str) -> pd.DataFrame:
    """Driver-spectrum rows for one alias-resolved cancer entity."""
    code = resolve_cancer_type(cancer_type)
    return (
        cancer_driver_spectrum_df()
        .loc[lambda df: df["cancer_code"].eq(code)]
        .reset_index(drop=True)
    )


def observed_driver_events(
    cancer_type: str, *, include_unresolved: bool = False
) -> tuple[str, ...]:
    """Distinct published driver events for an entity, preserving table order.

    ``include_unresolved=True`` also returns the explicit molecularly unresolved
    state. The result describes an entity spectrum, not a diagnosis requirement.
    """
    rows = cancer_driver_spectrum(cancer_type)
    if not include_unresolved:
        rows = rows.loc[~rows["driver_class"].eq("unresolved")]
    return tuple(dict.fromkeys(rows["driver_event"].dropna().astype(str)))
