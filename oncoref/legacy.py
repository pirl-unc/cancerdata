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

"""Machine-readable disposition of frozen legacy compatibility datasets."""

from __future__ import annotations

import pandas as pd

from .load_dataset import get_data

LEGACY_DATASET_STATUSES = frozenset(
    {
        "frozen_replacement_available",
        "frozen_downstream_owned",
    }
)

LEGACY_COMPATIBILITY_POLICIES = frozenset({"ship_read_only_no_new_rows"})


def legacy_dataset_dispositions() -> pd.DataFrame:
    """Return the reviewed migration/disposition row for every legacy table.

    Legacy tables remain importable for compatibility, but are frozen. The
    returned frame names the owner and replacement surface for new work, so
    callers do not need to infer the stack boundary from comments in the data
    manifest.
    """
    return get_data("legacy-dataset-dispositions").copy()


def legacy_dataset_disposition(dataset: str) -> dict | None:
    """Disposition record for one legacy dataset, or ``None`` if it is not legacy."""
    name = str(dataset).strip()
    rows = legacy_dataset_dispositions()
    hit = rows.loc[rows["dataset"].astype(str).eq(name)]
    return None if hit.empty else hit.iloc[0].to_dict()
