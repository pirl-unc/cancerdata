# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Discoverable immune-checkpoint-inhibitor response API.

This module is the organized public entry point for ICI response data:

- ``apd1_*`` helpers are the anti-PD-1 monotherapy compatibility slice.
- ``ici_*`` helpers cover the multi-regimen ICI tables.
- ``DEFAULT_ICI_REGIMEN_PRIORITY`` names the regimen preference order explicitly.

The older :mod:`oncoref.apd1` and :mod:`oncoref.ici` modules remain compatibility
facades for existing callers.
"""

from __future__ import annotations

from .apd1 import (
    cancer_apd1_response,
    cancer_apd1_response_df,
    cancer_apd1_response_record,
    resolve_apd1_response_source,
)
from .ici import (
    PROPORTION_METRICS,
    REGIMEN_FALLBACK,
    REGIMEN_LABELS,
    cancer_ici_regimen,
    cancer_ici_response,
    cancer_ici_response_df,
    cancer_ici_response_estimates_df,
    cancer_ici_response_record,
    cancer_ici_source_locator_audit_df,
    ici_regimens,
    pooled_ici_response,
    resolve_ici_response_source,
)

DEFAULT_ICI_REGIMEN_PRIORITY = REGIMEN_FALLBACK
"""Regimen priority for unpinned ICI response lookup: PD-1, then PD-L1, then combo."""

ICI_REGIMEN_LABELS = REGIMEN_LABELS
"""Human-readable labels for the ICI regimen tags."""

RESPONSE_PROPORTION_METRICS = PROPORTION_METRICS
"""Response metrics that can be responder-weighted pooled."""


def apd1_response(cancer_type=None, *, inherit: bool = True, include_inherited: bool = False):
    """Anti-PD-1 monotherapy ORR (%) for one cancer type, or the full code map."""
    return cancer_apd1_response(cancer_type, inherit=inherit, include_inherited=include_inherited)


def apd1_response_df():
    """Curated anti-PD-1 monotherapy ORR anchor table."""
    return cancer_apd1_response_df()


def apd1_response_record(
    cancer_type=None, *, inherit: bool = True, include_inherited: bool = False
):
    """Anti-PD-1 response with source/evidence metadata."""
    return cancer_apd1_response_record(
        cancer_type, inherit=inherit, include_inherited=include_inherited
    )


def apd1_response_source(cancer_type, *, inherit: bool = True):
    """Direct/proxy/ancestor anti-PD-1 evidence-source resolution."""
    return resolve_apd1_response_source(cancer_type, inherit=inherit)


def ici_response_anchor_df():
    """Curated representative ICI ORR anchor table, one row per cancer/regimen.

    Also carries the audited gap rows — codes with no defensible representative ORR,
    present with a blank ``orr_pct`` and ``regimen`` and an explicit ``missing_reason``.
    Filter on ``orr_pct.notna()`` for anchored values only."""
    return cancer_ici_response_df()


def ici_response_estimates_df():
    """Audited ICI endpoint evidence table with metrics, CIs, denominators, and refs."""
    return cancer_ici_response_estimates_df()


def ici_source_locator_audit_df():
    """Source-document locations and match evidence for every ICI estimate row."""
    return cancer_ici_source_locator_audit_df()


def best_available_ici_response(
    cancer_type=None, *, inherit: bool = True, include_inherited: bool = False
):
    """Best-available ICI ORR (%) using ``DEFAULT_ICI_REGIMEN_PRIORITY``.

    This is a clearer name for ``cancer_ici_response(..., regimen=None,
    fallback=True)``. It chooses anti-PD-1 monotherapy when present, then anti-PD-L1,
    then anti-PD-1 + anti-CTLA-4.
    """
    return cancer_ici_response(cancer_type, inherit=inherit, include_inherited=include_inherited)


def best_available_ici_response_record(
    cancer_type=None, *, inherit: bool = True, include_inherited: bool = False
):
    """Best-available ICI response with source/evidence metadata."""
    return cancer_ici_response_record(
        cancer_type, inherit=inherit, include_inherited=include_inherited
    )


def ici_response_by_regimen(
    cancer_type=None, *, inherit: bool = True, include_inherited: bool = False
):
    """ICI ORR values grouped by regimen instead of using regimen priority."""
    return cancer_ici_response(
        cancer_type,
        fallback=False,
        inherit=inherit,
        include_inherited=include_inherited,
    )


def ici_response_records_by_regimen(
    cancer_type=None, *, inherit: bool = True, include_inherited: bool = False
):
    """ICI response records grouped by regimen instead of using regimen priority."""
    return cancer_ici_response_record(
        cancer_type,
        fallback=False,
        inherit=inherit,
        include_inherited=include_inherited,
    )


def ici_response_source(cancer_type, *, regimen=None, fallback: bool = True, inherit: bool = True):
    """Direct/proxy/ancestor evidence-source resolution for one requested cancer type."""
    return resolve_ici_response_source(
        cancer_type, regimen=regimen, fallback=fallback, inherit=inherit
    )


def selected_ici_regimen(cancer_type, *, inherit: bool = True):
    """Regimen selected by ``best_available_ici_response`` for one cancer type.

    Uses the same direct/source-scope/ancestor resolution and ``inherit`` policy as
    the response lookup. Returns ``None`` for audited gaps and unresolved codes.
    """
    return resolve_ici_response_source(cancer_type, inherit=inherit)["selected_regimen"]


__all__ = [
    "DEFAULT_ICI_REGIMEN_PRIORITY",
    "ICI_REGIMEN_LABELS",
    "RESPONSE_PROPORTION_METRICS",
    "apd1_response",
    "apd1_response_df",
    "apd1_response_record",
    "apd1_response_source",
    "best_available_ici_response",
    "best_available_ici_response_record",
    "cancer_apd1_response",
    "cancer_apd1_response_df",
    "cancer_apd1_response_record",
    "cancer_ici_regimen",
    "cancer_ici_response",
    "cancer_ici_response_df",
    "cancer_ici_response_estimates_df",
    "cancer_ici_response_record",
    "cancer_ici_source_locator_audit_df",
    "ici_regimens",
    "ici_response_anchor_df",
    "ici_response_by_regimen",
    "ici_response_estimates_df",
    "ici_response_records_by_regimen",
    "ici_response_source",
    "ici_source_locator_audit_df",
    "pooled_ici_response",
    "resolve_apd1_response_source",
    "resolve_ici_response_source",
    "selected_ici_regimen",
]
