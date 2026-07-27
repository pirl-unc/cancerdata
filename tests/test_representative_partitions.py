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

import pandas as pd
import pytest

from oncoref.representative_partitions import (
    REPRESENTATIVE_PARTITION_POLICY_VERSION,
    assign_representative_partitions,
)


def _rows(cancer_code, groups, *, eligible=True):
    return [
        {
            "cancer_code": cancer_code,
            "representative_id": f"{cancer_code}__rep{rank}",
            "source_group_id": group_id,
            "benchmark_eligible": eligible,
        }
        for rank, group_id in enumerate(groups, start=1)
    ]


def test_five_group_cohort_gets_deterministic_three_two_split():
    provenance = pd.DataFrame(_rows("A", [f"P:s{i}" for i in range(5)]))

    first = assign_representative_partitions(provenance)
    second = assign_representative_partitions(provenance.sample(frac=1, random_state=3))

    first_roles = first.set_index("source_group_id")["partition_role"].to_dict()
    second_roles = second.set_index("source_group_id")["partition_role"].to_dict()
    assert first_roles == second_roles
    assert pd.Series(first_roles).value_counts().to_dict() == {
        "train": 3,
        "validation": 2,
    }
    assert set(first["partition_policy_version"]) == {REPRESENTATIVE_PARTITION_POLICY_VERSION}
    assert set(first["partition_status"]) == {"available"}
    assert set(first["partition_validation_target"]) == {2}


def test_shared_source_group_never_crosses_partition_roles():
    rows = _rows("PARENT", ["P:shared", "P:p2", "P:p3", "P:p4", "P:p5"])
    rows += _rows("SUBTYPE", ["P:shared", "P:s2"])

    partitioned = assign_representative_partitions(pd.DataFrame(rows))

    assert partitioned.groupby("source_group_id")["partition_role"].nunique().max() == 1
    subtype = partitioned[partitioned["cancer_code"] == "SUBTYPE"]
    assert set(subtype["partition_status"]) == {"available"}
    assert set(subtype["n_partition_train"]) == {1}
    assert set(subtype["n_partition_validation"]) == {1}


def test_single_group_cohort_is_explicitly_train_only():
    partitioned = assign_representative_partitions(pd.DataFrame(_rows("SPARSE", ["P:only"])))

    assert partitioned.loc[0, "partition_role"] == "train"
    assert partitioned.loc[0, "partition_status"] == "insufficient_independent_groups"
    assert partitioned.loc[0, "partition_validation_target"] == 0


def test_benchmark_ineligible_group_is_audit_only_for_every_alias():
    rows = _rows("A", ["P:shared"], eligible=True)
    rows += _rows("B", ["P:shared"], eligible=False)

    partitioned = assign_representative_partitions(pd.DataFrame(rows))

    assert set(partitioned["partition_role"]) == {"audit_only"}
    assert set(partitioned["partition_reason"]) == {"benchmark_ineligible_source_group"}
    assert set(partitioned["partition_status"]) == {"no_benchmark_eligible_groups"}


def test_independent_source_project_is_held_out_as_a_whole():
    rows = _rows("A", ["PRIMARY:p1", "PRIMARY:p2", "PRIMARY:p3"])
    rows += _rows("A", ["EXTERNAL:e1", "EXTERNAL:e2"])

    partitioned = assign_representative_partitions(pd.DataFrame(rows))
    external = partitioned[partitioned["source_group_id"].str.startswith("EXTERNAL:")]
    primary = partitioned[partitioned["source_group_id"].str.startswith("PRIMARY:")]

    assert len(set(external["partition_role"])) == 1
    held_out = external if external.iloc[0]["partition_role"] == "validation_external" else primary
    training = primary if held_out is external else external
    assert set(held_out["partition_role"]) == {"validation_external"}
    assert set(training["partition_role"]) == {"train"}
    assert set(partitioned["n_partition_validation_external"]) == {len(held_out)}


def test_partition_input_rejects_missing_source_namespace():
    provenance = pd.DataFrame(_rows("A", ["sample_without_namespace"]))

    with pytest.raises(ValueError, match="<source-namespace>:<source-sample>"):
        assign_representative_partitions(provenance)


def test_empty_partition_input_returns_stable_schema():
    provenance = pd.DataFrame(columns=["cancer_code", "source_group_id", "benchmark_eligible"])

    partitioned = assign_representative_partitions(provenance)

    assert partitioned.empty
    assert "partition_role" in partitioned
    assert "partition_status" in partitioned
