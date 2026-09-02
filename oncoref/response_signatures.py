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

"""Legacy compatibility surface for a small checkpoint-response signature set.

These helpers remain importable for existing callers and oncoref's historical
plots, but they are not the future home for therapy-response signature curation.
Purpose-specific response panels belong in pirlygenes, and this compatibility
surface should move there rather than grow in oncoref. oncoref should only own a
future response-signature table if it is recast as a source-anchored empirical
fact/provenance table rather than a marker-panel catalog.

``cancer-response-signatures.csv``: ``signature, gene_symbol, direction``
(``positive`` = response-associated, ``negative`` = resistance-associated),
``description``, ``source``. :func:`signature_score` scores a cohort by averaging
its member genes' (log) clean-TPM expression over the cohort's patients.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .legacy import warn_legacy_dataset_access
from .load_dataset import _register_derived_cache, get_data


@lru_cache(maxsize=1)
def _frame() -> pd.DataFrame:
    warn_legacy_dataset_access("cancer-response-signatures", stacklevel=3)
    return get_data("cancer-response-signatures", copy=False)


_register_derived_cache(_frame.cache_clear)


def response_signatures_df() -> pd.DataFrame:
    """Frozen response-signature compatibility snapshot; defensive copy.

    New purpose-specific therapy-response signature panels belong to
    pirlygenes. This snapshot remains available for historical oncoref plots.
    """
    return _frame().copy()


def response_signature_names() -> list[str]:
    """The available signature names (sorted)."""
    return sorted(_frame()["signature"].astype(str).unique())


def response_signature_genes(name: str) -> set[str]:
    """Member gene symbols of a signature. Raises if ``name`` is unknown."""
    df = _frame()
    sub = df[df["signature"].astype(str) == name]
    if sub.empty:
        raise ValueError(f"unknown signature {name!r}; one of {response_signature_names()}")
    return set(sub["gene_symbol"].astype(str))


def response_signature_direction(name: str) -> str:
    """``"positive"`` (response-associated) or ``"negative"`` (resistance) for a
    signature."""
    df = _frame()
    sub = df[df["signature"].astype(str) == name]
    if sub.empty:
        raise ValueError(f"unknown signature {name!r}; one of {response_signature_names()}")
    return str(sub["direction"].iloc[0])


def signature_score(cancer_type, name: str, *, statistic: str = "mean") -> float:
    """Score a cohort for a signature: the average (``statistic``) of its member
    genes' **log clean-TPM** cohort expression across the cohort's patients.

    Built on :func:`oncoref.expression.cohort_mean_expression` (so it needs the
    cohort's per-sample matrix). Genes absent from the cohort are skipped; ``nan`` if
    none of the signature's genes are present. Higher = more of that program; pair
    with :func:`response_signature_direction` for the response interpretation."""
    from .expression import cohort_mean_expression

    genes = response_signature_genes(name)
    mean = cohort_mean_expression(cancer_type, normalize="tpm_clean_log1p", statistic=statistic)
    vals = mean.loc[mean["Symbol"].astype(str).isin(genes), "expression"]
    return float(vals.mean()) if len(vals) else float("nan")


__all__ = [
    "response_signature_direction",
    "response_signature_genes",
    "response_signature_names",
    "response_signatures_df",
    "signature_score",
]
