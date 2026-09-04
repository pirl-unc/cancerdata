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

import numpy as np
import pandas as pd

from oncoref._evidence_resolution import evidence_record, parent_code, resolve_evidence


def _resolve(
    requested_code,
    *,
    values,
    gaps=None,
    sources=None,
    parents=None,
    inherit=True,
):
    gaps = gaps or {}
    sources = sources or {}
    return resolve_evidence(
        requested_code,
        direct_lookup=values.get,
        direct_gap_lookup=gaps.get,
        source_code_for=lambda code: sources.get(code, code),
        parent_by_code=parents or {},
        inherit=inherit,
    )


def test_direct_value_precedes_gap_and_all_inheritance_routes():
    resolution = _resolve(
        "CHILD",
        values={"CHILD": "direct", "SOURCE": "source", "PARENT": "ancestor"},
        gaps={"CHILD": "gap"},
        sources={"CHILD": "SOURCE"},
        parents={"CHILD": "PARENT"},
    )

    assert resolution.resolved_code == "CHILD"
    assert resolution.inheritance_kind == "direct"
    assert resolution.payload == "direct"


def test_direct_gap_blocks_source_and_ancestor_even_without_inheritance():
    kwargs = {
        "values": {"SOURCE": "source", "PARENT": "ancestor"},
        "gaps": {"CHILD": "reviewed gap"},
        "sources": {"CHILD": "SOURCE"},
        "parents": {"CHILD": "PARENT"},
    }

    for inherit in (True, False):
        resolution = _resolve("CHILD", inherit=inherit, **kwargs)
        assert resolution.resolved_code == "CHILD"
        assert resolution.inheritance_kind == "direct_missing"
        assert resolution.payload == "reviewed gap"


def test_source_scope_precedes_nearest_ancestor_and_inheritance_can_be_disabled():
    kwargs = {
        "values": {"SOURCE": "source", "PARENT": "ancestor"},
        "sources": {"CHILD": "SOURCE"},
        "parents": {"CHILD": "PARENT"},
    }

    inherited = _resolve("CHILD", **kwargs)
    assert inherited.resolved_code == "SOURCE"
    assert inherited.inheritance_kind == "source_scope"
    assert inherited.payload == "source"

    missing = _resolve("CHILD", inherit=False, **kwargs)
    assert missing.resolved_code == "CHILD"
    assert missing.inheritance_kind == "missing"
    assert missing.payload is None


def test_nearest_ancestor_wins_and_cycles_terminate_as_missing():
    nearest = _resolve(
        "CHILD",
        values={"PARENT": "nearest", "ROOT": "root"},
        parents={"CHILD": "PARENT", "PARENT": "ROOT"},
    )
    assert (nearest.resolved_code, nearest.inheritance_kind, nearest.payload) == (
        "PARENT",
        "ancestor",
        "nearest",
    )

    cycle = _resolve(
        "CHILD",
        values={},
        parents={"CHILD": "PARENT", "PARENT": "CHILD"},
    )
    assert cycle.inheritance_kind == "missing"
    assert cycle.payload is None


def test_nullable_parent_and_public_record_conversion_are_shared():
    assert parent_code("ROOT", {"ROOT": np.nan}) is None
    assert parent_code("CHILD", {"CHILD": " PARENT "}) == "PARENT"

    row = pd.Series({"missing": pd.NA, "count": np.int64(3), "label": "value"})
    record = evidence_record(
        row,
        requested_code="CHILD",
        resolved_code="PARENT",
        inheritance_kind="ancestor",
    )
    assert record == {
        "missing": None,
        "count": 3,
        "label": "value",
        "requested_cancer_code": "CHILD",
        "resolved_cancer_code": "PARENT",
        "inheritance_kind": "ancestor",
        "is_inherited_evidence": True,
    }
