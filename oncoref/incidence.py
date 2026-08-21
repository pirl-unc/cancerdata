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

"""Cancer incidence / mortality burden reference data and the registry-driven
resolution of a cancer type to its anatomic burden category."""

from __future__ import annotations

from functools import lru_cache

from .cancer_types import cancer_type_registry, resolve_cancer_type
from .load_dataset import _register_derived_cache, get_data

_BURDEN_METRICS = (
    "us_incidence_pct",
    "us_mortality_pct",
    "world_incidence_pct",
    "world_mortality_pct",
)


def cancer_burden_df():
    """Return the curated ``cancer-incidence-mortality.csv`` reference: each
    cancer **burden category**'s share (%) of annual cancer incidence and
    mortality, for the US and worldwide, cited per row (ACS Cancer Facts &
    Figures 2024 / GLOBOCAN 2022).

    The percentage columns are the current public lookup surface. The companion
    ``*_count`` / ``*_total`` columns, ``*_source_locator`` fields, source-site
    fields, ``derivation_basis``, and ``rounding_rule`` are the structured audit
    contract. ``aggregation`` and ``source_anchor`` preserve the complete legacy
    pirlygenes site-composition and resolvable citation surface so consumers can
    delegate without a compatibility table. Rows keep
    ``*_source_locator_status="not_extracted"`` until exact per-region exports and
    raw counts are filled in. Rows flagged in ``notes`` as subsets/rollups overlap
    others — don't sum them blindly."""
    return get_data("cancer-incidence-mortality")


def cancer_code_burden_map():
    """Return ``{cancer_code: burden_category}`` from
    ``cancer-code-burden-map.csv``. This is now only the small set of
    **overrides** the registry ontology can't express on its own (e.g.
    ``SARC_KS`` -> Kaposi rather than soft-tissue; ``LAML`` -> AML rather than
    other-leukemia; ``HL`` -> Hodgkin; ``CTCL`` -> non-Hodgkin). Everything else
    is resolved by :func:`burden_category` from the registry's family +
    primary_tissue."""
    df = get_data("cancer-code-burden-map")
    return dict(zip(df["cancer_code"].astype(str), df["burden_category"].astype(str)))


def cancer_burden(category=None, *, metric="us_incidence_pct"):
    """Burden share (%) for one category and metric, or the whole
    ``{category: pct}`` map. ``metric`` is one of ``us_incidence_pct``,
    ``us_mortality_pct``, ``world_incidence_pct``, ``world_mortality_pct``."""
    if metric not in _BURDEN_METRICS:
        raise ValueError(f"metric must be one of {_BURDEN_METRICS}")
    df = cancer_burden_df()
    mapping = dict(zip(df["burden_category"].astype(str), df[metric].astype(float)))
    if category is None:
        return mapping
    return mapping.get(category)


# Burden categories are anatomic-site shares (how ACS/GLOBOCAN tabulate), so a
# cohort is resolved to its category straight from the **cancer-type registry
# ontology** — one source of truth — rather than a parallel hand-map: the
# sarcoma family splits bone vs soft tissue on primary_tissue, plasma-cell and a
# handful of leukemia/lymphoma exceptions resolve by family, and primary tissue
# decides everything else. ``cancer-code-burden-map.csv`` now holds only the few
# true exceptions the ontology can't express.

# Sarcoma family -> bone_and_joint when its primary_tissue is skeletal, else
# soft_tissue_sarcoma. This (and heme-plasma -> myeloma) is a genuine family
# *override*: the histology decides the burden category even when the anatomic
# site would map elsewhere (e.g. DSRCT in peritoneum, DFSP in skin, endometrial
# stromal sarcoma) — so it is kept in code, not data. The bulk lookups
# (primary_tissue -> category, family -> category fallback) live in data files.
_BONE_SARCOMA_TISSUES = {"bone", "cartilage", "notochord"}


@lru_cache(maxsize=1)
def _tissue_burden_map() -> dict[str, str]:
    """``{primary_tissue: burden_category}`` from ``tissue-burden-map.csv`` (solid +
    heme tissues; the scopes don't share a tissue token so one flat map suffices)."""
    df = get_data("tissue-burden-map")
    return dict(zip(df["primary_tissue"].astype(str), df["burden_category"].astype(str)))


@lru_cache(maxsize=1)
def _family_burden_map() -> dict[str, str]:
    """``{family: burden_category}`` fallback from ``family-burden-map.csv`` — used
    only when a code's primary_tissue is blank/unmapped (melanoma at non-skin sites,
    the cns-* families, carcinoma-skin)."""
    df = get_data("family-burden-map")
    return dict(zip(df["family"].astype(str), df["burden_category"].astype(str)))


_register_derived_cache(_tissue_burden_map.cache_clear)
_register_derived_cache(_family_burden_map.cache_clear)


def burden_category(cancer_type):
    """Map a cancer type to its **incidence/mortality (cancer-burden) category** —
    the coarse anatomic bucket under which ACS / GLOBOCAN report population burden
    (e.g. ``SARC_OS`` → ``"bone_and_joint"``), so a fine cancer code can be joined to
    :func:`cancer_burden`. This is *not* a severity label; it is the granularity at
    which burden statistics are tabulated.

    Robustly resolves a cancer type (code, alias, or display name) to that category,
    driven by the cancer-type registry ontology.
    Order: normalize via :func:`resolve_cancer_type`; the small explicit
    ``cancer-code-burden-map`` *override* (walking the ``parent_code`` chain);
    then registry-driven — sarcoma family splits bone vs soft tissue, plasma
    cell -> myeloma, then ``primary_tissue`` (``tissue-burden-map.csv``), then
    ``family`` (``family-burden-map.csv``). Returns ``None`` only when nothing
    matches — callers should **warn**, not silently skip (an unmapped cohort is
    a coverage gap)."""
    try:
        code = resolve_cancer_type(cancer_type)
    except ValueError:
        return None
    if code is None:
        return None
    override = cancer_code_burden_map()
    tissue_map = _tissue_burden_map()
    family_map = _family_burden_map()
    reg = cancer_type_registry().set_index("code")
    # 1. explicit override (true exceptions only), walking up the parent chain
    cur, seen = code, set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur in override:
            return override[cur]
        if cur not in reg.index:
            break
        cur = str(reg.loc[cur].get("parent_code", "") or "").strip() or None
    # 2. registry-driven, walking up the parent chain for blank tissue/family
    cur, seen = code, set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur not in reg.index:
            break
        row = reg.loc[cur]
        family = str(row.get("family", "") or "")
        tissue = str(row.get("primary_tissue", "") or "")
        if family == "sarcoma":
            return "bone_and_joint" if tissue in _BONE_SARCOMA_TISSUES else "soft_tissue_sarcoma"
        if family == "heme-plasma":
            return "multiple_myeloma"
        if tissue in tissue_map:
            return tissue_map[tissue]
        if family in family_map:
            return family_map[family]
        cur = str(row.get("parent_code", "") or "").strip() or None
    return None
