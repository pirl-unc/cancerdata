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

"""Shared evidence-source resolution for cancer-code keyed reference tables."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

import pandas as pd

Payload = TypeVar("Payload")


@dataclass(frozen=True)
class EvidenceResolution(Generic[Payload]):
    """One completed direct/source-scope/ancestor evidence lookup."""

    resolved_code: str
    inheritance_kind: str
    payload: Payload | None


def parent_code(code: str, parent_by_code: Mapping[str, object]) -> str | None:
    """Return a clean parent code from a registry mapping, including nullable data."""
    parent = parent_by_code.get(code)
    if parent is None or pd.isna(parent):
        return None
    parent = str(parent).strip()
    return parent or None


def resolve_evidence(
    requested_code: str,
    *,
    direct_lookup: Callable[[str], Payload | None],
    source_code_for: Callable[[str], str],
    parent_by_code: Mapping[str, object],
    inherit: bool,
    direct_gap_lookup: Callable[[str], Payload | None] | None = None,
) -> EvidenceResolution[Payload]:
    """Resolve one evidence payload with a single, ordered inheritance contract.

    Resolution order is deliberately uniform across ICI, anti-PD-1, and TMB:

    1. a value curated directly for the requested code;
    2. an audited gap curated directly for that code;
    3. the registry's evidence-source code;
    4. the nearest ancestor with a value;
    5. missing.

    A direct audited gap is checked before ``inherit`` and therefore always blocks
    proxy or ancestor evidence. Disabling inheritance still permits direct values and
    direct gaps, but skips both source-scope and ancestor resolution.
    """
    direct = direct_lookup(requested_code)
    if direct is not None:
        return EvidenceResolution(requested_code, "direct", direct)

    if direct_gap_lookup is not None:
        gap = direct_gap_lookup(requested_code)
        if gap is not None:
            return EvidenceResolution(requested_code, "direct_missing", gap)

    if not inherit:
        return EvidenceResolution(requested_code, "missing", None)

    source_code = source_code_for(requested_code)
    if source_code != requested_code:
        source = direct_lookup(source_code)
        if source is not None:
            return EvidenceResolution(source_code, "source_scope", source)

    current = parent_code(requested_code, parent_by_code)
    seen = {requested_code}
    while current and current not in seen:
        seen.add(current)
        ancestor = direct_lookup(current)
        if ancestor is not None:
            return EvidenceResolution(current, "ancestor", ancestor)
        current = parent_code(current, parent_by_code)

    return EvidenceResolution(requested_code, "missing", None)


def public_value(value):
    """Convert pandas/numpy missing values and scalars for a public record."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def evidence_record(
    row,
    *,
    requested_code: str,
    resolved_code: str,
    inheritance_kind: str,
) -> dict[str, object]:
    """Convert a resolved pandas row to the common public record contract."""
    record = {key: public_value(row[key]) for key in row.index}
    record["requested_cancer_code"] = requested_code
    record["resolved_cancer_code"] = resolved_code
    record["inheritance_kind"] = inheritance_kind
    record["is_inherited_evidence"] = requested_code != resolved_code
    return record
