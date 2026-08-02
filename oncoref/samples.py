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

"""Per-sample curation manifest for the cohort expression references.

``cancer-reference-expression-samples.csv.gz`` records, for every individual
sample that went into a cohort summary, its provenance and the curation decision
(which cancer code it was assigned, which cohort/project it came from, the source
file, the raw unit + processing pipeline, and whether it was included or excluded
— with the exclusion reason). This is the sample-level curation needed to
reproduce / audit the per-cohort summaries.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .cancer_types import resolve_cancer_type
from .load_dataset import get_data


@lru_cache(maxsize=1)
def _manifest() -> pd.DataFrame:
    return get_data("cancer-reference-expression-samples", copy=False)


def sample_manifest() -> pd.DataFrame:
    """The full per-sample curation manifest (one row per sample). Defensive copy."""
    return _manifest().copy()


def samples_for_cancer_code(code: str, *, included_only: bool = True) -> pd.DataFrame:
    """Manifest rows assigned to a cancer code (included samples by default).

    The code is alias-resolved, so ``"prostate"`` / ``"lung_adeno"`` / a pre-rename
    code all work (an unknown code passes through and simply matches nothing)."""
    resolved = resolve_cancer_type(code, strict=False) or code
    df = _manifest()
    sub = df[df["cancer_code"].astype(str) == resolved]
    if included_only and "included" in sub.columns:
        sub = sub[sub["included"].astype(str).str.lower().isin(("true", "1"))]
    return sub.copy()


def samples_for_cohort(source_cohort: str, *, included_only: bool = True) -> pd.DataFrame:
    """Manifest rows from a source cohort (included samples by default)."""
    df = _manifest()
    sub = df[df["source_cohort"].astype(str) == source_cohort]
    if included_only and "included" in sub.columns:
        sub = sub[sub["included"].astype(str).str.lower().isin(("true", "1"))]
    return sub.copy()


def sample_counts_by_cancer_code(*, included_only: bool = True) -> pd.Series:
    """Number of curated samples per cancer code (a quick coverage view)."""
    df = _manifest()
    if included_only and "included" in df.columns:
        df = df[df["included"].astype(str).str.lower().isin(("true", "1"))]
    return df["cancer_code"].astype(str).value_counts()


@lru_cache(maxsize=1)
def _molecular_provenance() -> pd.DataFrame:
    return get_data("cancer-reference-sample-molecular-provenance", copy=False)


def sample_molecular_provenance() -> pd.DataFrame:
    """Public diagnosis and molecular evidence for reference/acquisition samples.

    Rows are libraries. ``donor_id`` makes technical or longitudinal libraries
    auditable without treating them as independent donors. A diagnosis-only row
    has ``molecular_status="unknown"`` and ``orthogonal_confirmed=False``;
    diagnosis never supplies a fusion call.
    """
    return _molecular_provenance().copy()


def molecular_provenance_for_cancer_code(code: str) -> pd.DataFrame:
    """Sample-level molecular provenance for one alias-resolved cancer code."""
    resolved = resolve_cancer_type(code, strict=False) or code
    rows = _molecular_provenance()
    return rows.loc[rows["cancer_code"].astype(str).eq(resolved)].reset_index(drop=True).copy()


def molecular_provenance_for_sample(sample_id: str) -> pd.DataFrame:
    """All provenance rows matching a sample, library, or donor identifier."""
    query = str(sample_id)
    rows = _molecular_provenance()
    mask = (
        rows["sample_id"].astype(str).eq(query)
        | rows["library_id"].astype(str).eq(query)
        | rows["donor_id"].astype(str).eq(query)
    )
    return rows.loc[mask].reset_index(drop=True).copy()


def molecular_sample_counts(code: str | None = None) -> pd.DataFrame:
    """Library and distinct-donor counts by physical source cohort.

    The optional cancer code is alias-resolved. Counts include public expression
    and controlled acquisition rows; ``n_expression_libraries`` states how many
    of the listed libraries currently have loadable expression.
    """
    rows = _molecular_provenance() if code is None else molecular_provenance_for_cancer_code(code)
    group_cols = ["cancer_code", "source_id", "source_cohort", "access_level"]
    if rows.empty:
        return pd.DataFrame(
            columns=[*group_cols, "n_libraries", "n_donors", "n_expression_libraries"]
        )
    counts = (
        rows.groupby(group_cols, dropna=False, sort=True)
        .agg(
            n_libraries=("library_id", "nunique"),
            n_donors=("donor_id", "nunique"),
            n_expression_libraries=("expression_available", "sum"),
        )
        .reset_index()
    )
    return counts
