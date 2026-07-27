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

"""Leakage-resistant train/validation roles for representative expression vectors."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import pandas as pd

REPRESENTATIVE_PARTITION_POLICY_VERSION = "source_grouped_3_train_2_validation_v1"
REPRESENTATIVE_PARTITION_ROLES = (
    "train",
    "validation",
    "validation_external",
    "audit_only",
)
REPRESENTATIVE_PARTITION_COHORT_COLUMNS = (
    "cancer_code",
    "partition_policy_version",
    "partition_validation_target",
    "partition_status",
    "n_partition_train",
    "n_partition_validation",
    "n_partition_validation_external",
    "n_partition_audit_only",
)

_REQUIRED_COLUMNS = {
    "cancer_code",
    "source_group_id",
    "benchmark_eligible",
}


def _stable_order(value: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{REPRESENTATIVE_PARTITION_POLICY_VERSION}:{value}".encode()
    ).hexdigest()
    return digest, value


def _source_project(source_group_id: str) -> str:
    """Return the stable source namespace embedded in a source-group ID."""

    return source_group_id.split(":", 1)[0]


def _validation_target(n_groups: int) -> int:
    """Hold out up to two groups, never more than half of a cohort."""

    return min(2, n_groups // 2)


def _group_memberships(cohort_groups: dict[str, set[str]]) -> dict[str, set[str]]:
    memberships: dict[str, set[str]] = defaultdict(set)
    for cancer_code, groups in cohort_groups.items():
        for group_id in groups:
            memberships[group_id].add(cancer_code)
    return dict(memberships)


def _validation_count(groups: set[str], roles: dict[str, str]) -> int:
    return sum(roles[group_id].startswith("validation") for group_id in groups)


def _can_hold_out(
    candidate_groups: set[str],
    *,
    cohort_groups: dict[str, set[str]],
    memberships: dict[str, set[str]],
    roles: dict[str, str],
    enforce_targets: bool,
) -> bool:
    affected_cohorts = {
        cancer_code for group_id in candidate_groups for cancer_code in memberships[group_id]
    }
    for cancer_code in affected_cohorts:
        groups = cohort_groups[cancer_code]
        held_out = {
            group_id
            for group_id in groups
            if roles[group_id].startswith("validation") or group_id in candidate_groups
        }
        if len(held_out) >= len(groups):
            return False
        if enforce_targets and len(held_out) > _validation_target(len(groups)):
            return False
    return True


def _assign_external_projects(
    *,
    cohort_groups: dict[str, set[str]],
    memberships: dict[str, set[str]],
    roles: dict[str, str],
) -> None:
    """Hold out one complete source project when a cohort spans independent projects."""

    for cancer_code in sorted(cohort_groups):
        groups = cohort_groups[cancer_code]
        by_project: dict[str, set[str]] = defaultdict(set)
        for group_id in groups:
            by_project[_source_project(group_id)].add(group_id)
        if len(by_project) < 2:
            continue

        projects = sorted(
            by_project,
            key=lambda project: (len(by_project[project]), _stable_order(project)),
        )
        for project in projects:
            candidate_groups = by_project[project]
            if not _can_hold_out(
                candidate_groups,
                cohort_groups=cohort_groups,
                memberships=memberships,
                roles=roles,
                enforce_targets=False,
            ):
                continue
            for group_id in candidate_groups:
                roles[group_id] = "validation_external"
            break


def _assign_stratified_validation(
    *,
    cohort_groups: dict[str, set[str]],
    memberships: dict[str, set[str]],
    roles: dict[str, str],
) -> None:
    """Fill each cohort's validation target without splitting shared source groups."""

    cohort_order = sorted(cohort_groups, key=lambda code: (len(cohort_groups[code]), code))
    for cancer_code in cohort_order:
        groups = cohort_groups[cancer_code]
        needed = _validation_target(len(groups)) - _validation_count(groups, roles)
        candidates = sorted(
            (group_id for group_id in groups if roles[group_id] == "train"),
            key=_stable_order,
        )
        for group_id in candidates:
            if needed <= 0:
                break
            if not _can_hold_out(
                {group_id},
                cohort_groups=cohort_groups,
                memberships=memberships,
                roles=roles,
                enforce_targets=True,
            ):
                continue
            roles[group_id] = "validation"
            needed -= 1


def _cohort_partition_summary(
    cancer_code: str,
    *,
    all_cohort_groups: dict[str, set[str]],
    cohort_groups: dict[str, set[str]],
    roles: dict[str, str],
) -> dict[str, int | str]:
    all_groups = all_cohort_groups[cancer_code]
    eligible_groups = cohort_groups.get(cancer_code, set())
    target = _validation_target(len(eligible_groups))
    n_train = sum(roles[group_id] == "train" for group_id in eligible_groups)
    n_external = sum(roles[group_id] == "validation_external" for group_id in eligible_groups)
    n_validation = _validation_count(eligible_groups, roles)
    n_audit = len(all_groups - eligible_groups)

    if not eligible_groups:
        status = "no_benchmark_eligible_groups"
    elif target == 0:
        status = "insufficient_independent_groups"
    elif n_validation >= target:
        status = "available"
    else:
        status = "shared_group_constraints"

    return {
        "partition_validation_target": target,
        "partition_status": status,
        "n_partition_train": n_train,
        "n_partition_validation": n_validation,
        "n_partition_validation_external": n_external,
        "n_partition_audit_only": n_audit,
    }


def assign_representative_partitions(provenance: pd.DataFrame) -> pd.DataFrame:
    """Annotate representative provenance with stable, source-grouped split roles.

    A physical ``source_group_id`` receives exactly one role across every parent,
    subtype, and alias row. Benchmark-ineligible groups are audit-only. Eligible
    cohorts hold out up to two groups, never more than half; a cohort with only one
    independent group remains train-only and is marked as insufficient.
    """

    missing = sorted(_REQUIRED_COLUMNS - set(provenance.columns))
    if missing:
        raise ValueError(f"representative partition input lacks columns: {missing}")

    out = provenance.copy()
    source_group_ids = out["source_group_id"].astype("string").str.strip()
    invalid_group_ids = (
        source_group_ids.isna() | source_group_ids.eq("") | ~source_group_ids.str.contains(":")
    )
    if invalid_group_ids.any():
        raise ValueError(
            "representative partition input has an invalid source_group_id; "
            "expected '<source-namespace>:<source-sample>'"
        )
    cancer_codes = out["cancer_code"].astype("string").str.strip()
    if (cancer_codes.isna() | cancer_codes.eq("")).any():
        raise ValueError("representative partition input has blank cancer_code")
    out["source_group_id"] = source_group_ids
    out["cancer_code"] = cancer_codes

    benchmark = out["benchmark_eligible"].astype("boolean")
    if benchmark.isna().any():
        raise ValueError("representative partition input has unknown benchmark_eligible")
    if out.empty:
        out["partition_role"] = pd.Series(dtype="string")
        out["partition_reason"] = pd.Series(dtype="string")
        out["partition_policy_version"] = pd.Series(dtype="string")
        for column in REPRESENTATIVE_PARTITION_COHORT_COLUMNS[2:]:
            out[column] = pd.Series(dtype="object")
        return out

    group_eligible = benchmark.groupby(out["source_group_id"]).all().to_dict()
    all_cohort_groups = {
        str(code): set(group["source_group_id"])
        for code, group in out.groupby("cancer_code", sort=True)
    }
    cohort_groups = {
        code: {group_id for group_id in groups if group_eligible[group_id]}
        for code, groups in all_cohort_groups.items()
    }
    cohort_groups = {code: groups for code, groups in cohort_groups.items() if groups}
    memberships = _group_memberships(cohort_groups)

    roles = {
        group_id: ("train" if eligible else "audit_only")
        for group_id, eligible in group_eligible.items()
    }
    _assign_external_projects(
        cohort_groups=cohort_groups,
        memberships=memberships,
        roles=roles,
    )
    _assign_stratified_validation(
        cohort_groups=cohort_groups,
        memberships=memberships,
        roles=roles,
    )

    summaries = {
        cancer_code: _cohort_partition_summary(
            cancer_code,
            all_cohort_groups=all_cohort_groups,
            cohort_groups=cohort_groups,
            roles=roles,
        )
        for cancer_code in all_cohort_groups
    }
    out["partition_role"] = out["source_group_id"].map(roles)
    out["partition_reason"] = out["partition_role"].map(
        {
            "train": "stable_source_group_training",
            "validation": "stable_source_group_validation",
            "validation_external": "independent_source_project_holdout",
            "audit_only": "benchmark_ineligible_source_group",
        }
    )
    out["partition_policy_version"] = REPRESENTATIVE_PARTITION_POLICY_VERSION
    for column in next(iter(summaries.values())):
        out[column] = out["cancer_code"].map(
            {code: summary[column] for code, summary in summaries.items()}
        )
    return out


def representative_partition_cohort_metadata(partitioned: pd.DataFrame) -> pd.DataFrame:
    """Return one partition-coverage row per cohort from annotated provenance."""

    missing = sorted(set(REPRESENTATIVE_PARTITION_COHORT_COLUMNS) - set(partitioned.columns))
    if missing:
        raise ValueError(f"partitioned representative provenance lacks columns: {missing}")
    cohort_metadata = partitioned[list(REPRESENTATIVE_PARTITION_COHORT_COLUMNS)].drop_duplicates()
    if cohort_metadata["cancer_code"].duplicated().any():
        raise ValueError("partitioned representative provenance has conflicting cohort metadata")
    return cohort_metadata.reset_index(drop=True)


def representative_partition_build_metadata(partitioned: pd.DataFrame) -> dict:
    """Return the bundle-level partition policy and unique source-group role counts."""

    required = {"source_group_id", "partition_role"}
    missing = sorted(required - set(partitioned.columns))
    if missing:
        raise ValueError(f"partitioned representative provenance lacks columns: {missing}")
    group_roles = partitioned[list(required)].drop_duplicates()
    if group_roles["source_group_id"].duplicated().any():
        raise ValueError("one representative source group has conflicting partition roles")
    role_counts = group_roles["partition_role"].value_counts().sort_index().astype(int).to_dict()
    return {
        "grouping_key": "source_group_id",
        "policy_version": REPRESENTATIVE_PARTITION_POLICY_VERSION,
        "role_counts": role_counts,
    }
