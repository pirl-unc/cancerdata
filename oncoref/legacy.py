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

import threading
import warnings

import pandas as pd

from .load_dataset import get_data

LEGACY_DATASET_STATUSES = frozenset(
    {
        "frozen_replacement_available",
        "frozen_migration_pending",
        "frozen_downstream_owned",
    }
)

LEGACY_COMPATIBILITY_POLICIES = frozenset({"ship_read_only_no_new_rows"})

LEGACY_TYPED_ACCESSORS = {
    "cancer-driver-genes": "oncoref.cancer_genes.cancer_driver_genes_df",
    "cancer-driver-variants": "oncoref.cancer_genes.cancer_driver_variants_df",
    "cancer-key-genes": "oncoref.cancer_genes.cancer_key_genes_df",
    "cancer-type-genes": "oncoref.cancer_genes.cancer_type_genes_df",
    "cancer-viral-antigens": "oncoref.cancer_genes.cancer_viral_antigens_df",
    "disease-state-rules": "oncoref.cancer_genes.disease_state_rules_df",
    "fusion-expression-effects": "oncoref.fusions.fusion_expression_effect_rules_df",
    "fusion-surrogate-expression": "oncoref.fusions.fusion_surrogate_expression_df",
    "narrative-gene-sets": "oncoref.cancer_genes.narrative_gene_sets_df",
    "cancer-response-signatures": "oncoref.response_signatures.response_signatures_df",
    "rare-cancer-fusion-rules": "oncoref.fusions.rare_cancer_fusion_rules_df",
}

_WARNED_LEGACY_DATASETS: set[str] = set()
_WARNING_LOCK = threading.Lock()


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


def warn_legacy_dataset_access(dataset: str, *, stacklevel: int = 2) -> None:
    """Warn once per process when a typed API reads a frozen snapshot.

    The generic :func:`oncoref.load_dataset.get_data` loader intentionally stays
    warning-free as a low-level compatibility escape hatch. Typed accessors call
    this helper so their owner, replacement surface, and freeze policy are not
    mistaken for current curation.
    """
    name = str(dataset).strip()
    disposition = legacy_dataset_disposition(name)
    if disposition is None:
        raise ValueError(f"{name!r} is not a registered legacy dataset")
    with _WARNING_LOCK:
        if name in _WARNED_LEGACY_DATASETS:
            return
        warnings.warn(
            f"{name!r} is a frozen compatibility snapshot "
            f"({disposition['current_status']}); new work belongs to "
            f"{disposition['replacement_owner']} via "
            f"{disposition['replacement_surface']}. "
            f"Policy: {disposition['compatibility_policy']}.",
            DeprecationWarning,
            stacklevel=stacklevel + 1,
        )
        _WARNED_LEGACY_DATASETS.add(name)
