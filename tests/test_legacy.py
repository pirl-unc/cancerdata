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

import oncoref
from oncoref import data_manifest, legacy


def _legacy_manifest_names():
    wheel = {**data_manifest.WHEEL, **data_manifest.CANCERDATA_ORIGINATED}
    return {name for name, (category, _) in wheel.items() if category == "legacy-compat"}


def test_every_legacy_compat_dataset_has_one_reviewed_disposition():
    rows = legacy.legacy_dataset_dispositions()

    assert rows["dataset"].is_unique
    assert set(rows["dataset"]) == _legacy_manifest_names()
    assert len(rows) == 11
    assert set(rows["current_status"]) <= legacy.LEGACY_DATASET_STATUSES
    assert set(rows["compatibility_policy"]) <= legacy.LEGACY_COMPATIBILITY_POLICIES
    for column in (
        "replacement_owner",
        "replacement_surface",
        "compatibility_policy",
        "rationale",
    ):
        assert rows[column].notna().all()
        assert rows[column].astype(str).str.strip().ne("").all()


def test_driver_legacy_tables_use_source_anchored_spectrum_for_new_facts():
    for dataset in ("cancer-driver-genes", "cancer-driver-variants"):
        row = legacy.legacy_dataset_disposition(dataset)
        assert row is not None
        assert row["current_status"] == "frozen_replacement_available"
        assert row["replacement_owner"] == "oncoref"
        assert row["replacement_surface"] == "cancer-entity-driver-spectrum"

    assert legacy.legacy_dataset_disposition("cancer-tmb") is None
    assert oncoref.legacy_dataset_dispositions is legacy.legacy_dataset_dispositions
