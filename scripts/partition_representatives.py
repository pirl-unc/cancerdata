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

"""Add the released representative partition contract to an existing data bundle.

This migration changes only representative provenance and expression build
metadata. Expression vectors remain byte-for-byte unchanged.

Run:
    python scripts/partition_representatives.py --bundle /path/to/bundle-staging
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.expression import EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION
from oncoref.representative_partitions import (
    REPRESENTATIVE_PARTITION_COHORT_COLUMNS,
    assign_representative_partitions,
    representative_partition_build_metadata,
    representative_partition_cohort_metadata,
)

_REPRESENTATIVE_DIR = "cancer-reference-expression-representatives"
_PROVENANCE_NAME = "_provenance.csv"
_COHORT_METADATA_NAME = "expression-artifact-build-metadata.csv"
_BUILD_METADATA_NAME = "expression-artifact-build-metadata.json"


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def partition_existing_bundle(bundle: Path) -> pd.DataFrame:
    """Partition existing representative provenance and update its build metadata."""

    provenance_path = bundle / _REPRESENTATIVE_DIR / _PROVENANCE_NAME
    cohort_metadata_path = bundle / _COHORT_METADATA_NAME
    build_metadata_path = bundle / _BUILD_METADATA_NAME
    for path in (provenance_path, cohort_metadata_path, build_metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"required bundle metadata is missing: {path}")

    provenance = pd.read_csv(provenance_path)
    if "cancer_code" not in provenance:
        provenance["cancer_code"] = (
            provenance["representative_id"]
            .astype(str)
            .str.replace(
                r"__rep\d+$",
                "",
                regex=True,
            )
        )
    partitioned = assign_representative_partitions(provenance)

    cohort_metadata = pd.read_csv(cohort_metadata_path)
    partition_columns = set(REPRESENTATIVE_PARTITION_COHORT_COLUMNS) - {"cancer_code"}
    cohort_metadata = cohort_metadata.drop(
        columns=[column for column in partition_columns if column in cohort_metadata],
    )
    cohort_metadata = cohort_metadata.merge(
        representative_partition_cohort_metadata(partitioned),
        on="cancer_code",
        how="left",
        validate="one_to_one",
    )

    build_metadata = json.loads(build_metadata_path.read_text())
    build_metadata["schema_version"] = EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION
    build_metadata["representative_partition"] = representative_partition_build_metadata(
        partitioned
    )

    _atomic_csv(partitioned, provenance_path)
    _atomic_csv(cohort_metadata, cohort_metadata_path)
    _atomic_json(build_metadata, build_metadata_path)
    return partitioned


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="Existing bundle staging root")
    args = parser.parse_args(argv)
    partitioned = partition_existing_bundle(args.bundle.expanduser())
    group_roles = partitioned.drop_duplicates("source_group_id")["partition_role"]
    print(
        f"partitioned {len(group_roles)} source groups across "
        f"{partitioned['cancer_code'].nunique()} cohorts: "
        + ", ".join(
            f"{role}={count}" for role, count in group_roles.value_counts().sort_index().items()
        )
    )


if __name__ == "__main__":
    main()
