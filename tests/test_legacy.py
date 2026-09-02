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

import importlib
import inspect
import warnings

import oncoref
from oncoref import catalog, data_manifest, legacy
from oncoref.load_dataset import _clear_cache, get_data


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


def test_driver_legacy_tables_use_complete_source_anchored_replacements():
    replacements = {
        "cancer-driver-genes": "driver-gene-evidence",
        "cancer-driver-variants": "driver-variant-evidence",
    }
    for dataset, replacement in replacements.items():
        row = legacy.legacy_dataset_disposition(dataset)
        assert row is not None
        assert row["current_status"] == "frozen_replacement_available"
        assert row["replacement_owner"] == "oncoref"
        assert row["replacement_surface"] == replacement

    assert legacy.legacy_dataset_disposition("cancer-tmb") is None
    assert oncoref.legacy_dataset_dispositions is legacy.legacy_dataset_dispositions


def _load_accessor(path):
    module_name, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), name)


def test_every_legacy_typed_accessor_warns_once_and_discloses_frozen_status():
    assert set(legacy.LEGACY_TYPED_ACCESSORS) == _legacy_manifest_names()
    _clear_cache()
    legacy._WARNED_LEGACY_DATASETS.clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            for _dataset, accessor_path in legacy.LEGACY_TYPED_ACCESSORS.items():
                accessor = _load_accessor(accessor_path)
                assert "frozen" in inspect.getdoc(accessor).lower()
                accessor()
                accessor()

        emitted = [item for item in caught if item.category is DeprecationWarning]
        assert len(emitted) == len(legacy.LEGACY_TYPED_ACCESSORS)
        messages = [str(item.message) for item in emitted]
        for dataset in legacy.LEGACY_TYPED_ACCESSORS:
            assert sum(dataset in message for message in messages) == 1
    finally:
        legacy._WARNED_LEGACY_DATASETS.clear()
        _clear_cache()


def test_catalog_inventory_discloses_legacy_policy_and_accessor():
    rows = {row["name"]: row for row in catalog.inventory()}
    for dataset, accessor in legacy.LEGACY_TYPED_ACCESSORS.items():
        disposition = legacy.legacy_dataset_disposition(dataset)
        row = rows[dataset]
        assert row["current_status"] == disposition["current_status"]
        assert row["replacement_owner"] == disposition["replacement_owner"]
        assert row["replacement_surface"] == disposition["replacement_surface"]
        assert row["compatibility_policy"] == disposition["compatibility_policy"]
        assert row["typed_accessor"] == accessor

    current = rows["cancer-fusions"]
    assert current["current_status"] is None
    assert current["replacement_owner"] is None
    assert current["replacement_surface"] is None
    assert current["compatibility_policy"] is None
    assert current["typed_accessor"] is None


def test_raw_get_data_remains_a_warning_free_compatibility_escape_hatch():
    legacy._WARNED_LEGACY_DATASETS.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert not get_data("cancer-driver-genes").empty
    assert not caught
