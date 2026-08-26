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

"""One catalog over every managed (fetchable) dataset.

oncoref holds two kinds of heavy/external data behind separate machinery: the
version-pinned per-cohort **expression bundle** (:mod:`oncoref.data_bundle`, a
single GitHub-release tarball) and the **HPA** protein/RNA reference tables
(:mod:`oncoref.reference_data`, per-source downloads from proteinatlas.org).
This module is the unified facade over both — one ``Dataset`` model and one
fetch/cache/status API — so a caller (or the CLI) manages everything uniformly
without knowing which backend a dataset lives in.

    from oncoref import catalog
    catalog.datasets()                       # every managed dataset
    catalog.status()                         # uniform present/size rows
    catalog.ensure("hpa_normal_tissue")      # download if needed -> Path
    catalog.ensure("cancer-reference-expression-percentiles")
    catalog.fetch("all")                     # materialize everything

The small curated tables (registry, TMB, fusions, …) ship in the wheel and need
no management, so they are intentionally not in the catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import data_bundle, data_manifest, reference_data, source_matrices

_BUNDLE = "bundle"
_HPA = "hpa"

#: prefix for addressing one cohort's raw per-sample matrix (``per-sample:LUAD``).
_PER_SAMPLE = "per-sample:"

#: Human-readable descriptions for the expression-bundle members.
_BUNDLE_DESCRIPTIONS = {
    "cancer-reference-expression": "per-cohort RNA-seq expression shards",
    "cancer-reference-expression-representatives": "per-cohort medoid/exemplar samples",
    "cancer-reference-expression-percentiles": "per-gene tail-weighted percentile vectors",
    "cancer-reference-expression-within-sample-top5": (
        "within-sample high-expression prevalence by cohort"
    ),
    "cancer-reference-expression-percentiles-proteoform-cta": (
        "CTA-scope proteoform-summed percentile vectors"
    ),
    "cancer-reference-expression-within-sample-top5-proteoform-cta": (
        "CTA-scope proteoform-summed within-sample prevalence"
    ),
    "source-matrix-sample-qc.csv": "per-sample QC manifest for rebuilt expression artifacts",
    "expression-artifact-build-metadata.csv": (
        "per-cohort build/QC provenance for rebuilt expression artifacts"
    ),
    "expression-artifact-build-metadata.json": (
        "bundle-level build/QC provenance for rebuilt expression artifacts"
    ),
    "pan-cancer-expression.csv": "pan-cancer HPA-tissue + TCGA expression matrix",
    "hpa-cell-type-expression.csv": "HPA cell-type nTPM matrix",
}


@dataclass(frozen=True)
class Dataset:
    """A managed dataset: its ``name``, the backend ``kind`` (``"bundle"`` or
    ``"hpa"``), and a one-line ``description``."""

    name: str
    kind: str
    description: str


def datasets() -> list[Dataset]:
    """Every managed dataset across both backends (bundle members + HPA sources)."""
    bundle_names = list(data_bundle.DOWNLOADABLE_PATHS)
    hpa_names = list(reference_data.REFERENCE_SOURCES)
    collisions = set(bundle_names) & set(hpa_names)
    if collisions:
        # The catalog routes by name; a name in both backends would silently
        # shadow one of them. Fail loudly instead of mis-routing.
        raise RuntimeError(f"dataset name collision across backends: {sorted(collisions)}")
    out = [
        Dataset(name, _BUNDLE, _BUNDLE_DESCRIPTIONS.get(name, "expression bundle artifact"))
        for name in bundle_names
    ]
    out += [
        Dataset(name, _HPA, reference_data.REFERENCE_SOURCES[name]["description"])
        for name in hpa_names
    ]
    return out


def _by_name() -> dict[str, Dataset]:
    return {d.name: d for d in datasets()}


def dataset(name: str) -> Dataset:
    """The :class:`Dataset` for ``name``; raises ``KeyError`` if unknown."""
    try:
        return _by_name()[name]
    except KeyError:
        avail = ", ".join(sorted(_by_name()))
        raise KeyError(f"unknown dataset {name!r}; available: {avail}") from None


def path(name: str) -> Path | None:
    """The dataset's on-disk location if present in the cache, else ``None``
    (never triggers a download). ``per-sample:<code>`` addresses one cohort's
    raw per-sample matrix."""
    if name.startswith(_PER_SAMPLE):
        code = name[len(_PER_SAMPLE) :]
        return source_matrices.local_path(code) if source_matrices.is_cached(code) else None
    d = dataset(name)
    if d.kind == _HPA:
        p = reference_data.local_path(name)
        return p if p.exists() else None
    return data_bundle.find(name)


def _per_sample_status_row(name: str) -> dict:
    code = name[len(_PER_SAMPLE) :]
    try:
        info = source_matrices.cohort_info(code)
        p = path(name)
    except source_matrices.SourceMatrixError as e:
        raise KeyError(str(e)) from None
    resolved = str(info["cancer_code"])
    return {
        "name": f"{_PER_SAMPLE}{resolved}",
        "kind": "source",
        "present": p is not None,
        "path": str(p) if p is not None else None,
        "size_bytes": _size_bytes(p),
        "cohorts": 1,
        "description": (
            f"raw per-sample TPM matrix for {resolved} "
            f"({info['source_cohort']}; n={int(info['n_samples'])})"
        ),
    }


def ensure(name: str) -> Path:
    """Local path to ``name``, downloading if absent. Returns the path.

    HPA sources fetch per-file; a bundle member triggers the one-time tarball
    download (all bundle members arrive together); ``per-sample:<code>`` fetches
    one cohort's raw matrix.
    """
    if name.startswith(_PER_SAMPLE):
        return source_matrices.ensure(name[len(_PER_SAMPLE) :])
    d = dataset(name)
    if d.kind == _HPA:
        return reference_data.ensure(name)
    data_bundle.ensure_local()
    resolved = data_bundle.find(name)
    if resolved is None:
        raise FileNotFoundError(
            f"{name!r} missing from the expression bundle after fetch ({data_bundle.cache_dir()})"
        )
    return resolved


def fetch(target: str = "all", *, force: bool = False) -> list[str]:
    """Materialize data, downloading what's missing. ``target`` selects what:

    - ``"all"`` — the reference data (expression bundle + every HPA source);
    - ``"bundle"`` / ``"hpa"`` — just that kind (e.g. ``fetch("hpa")`` for HPA only);
    - ``"source"`` — every cohort's raw per-sample matrix (~21 GB; opt-in, not in ``"all"``);
    - ``"per-sample:<code>"`` — one cohort's raw matrix;
    - a dataset name — just that dataset.

    Returns the names actually downloaded (already-cached skipped unless ``force``).
    The bundle tarball is fetched at most once.
    """
    # one cohort's raw per-sample matrix
    if target.startswith(_PER_SAMPLE):
        code = target[len(_PER_SAMPLE) :]
        already = source_matrices.is_cached(code)
        source_matrices.fetch(code, force=force)
        return [target] if force or not already else []
    # every per-sample matrix (heavy; deliberately not part of "all")
    if target == "source":
        out = []
        for code in source_matrices.available_cohorts():
            if force or not source_matrices.is_cached(code):
                source_matrices.fetch(code, force=force)
                out.append(f"{_PER_SAMPLE}{code}")
        return out

    if target == "all":
        targets = [d.name for d in datasets()]
    elif target in (_BUNDLE, _HPA):
        targets = [d.name for d in datasets() if d.kind == target]
    else:
        targets = [dataset(target).name]

    downloaded = []
    # The bundle is one tarball — fetch it a single time if any member is targeted.
    bundle_targets = [n for n in targets if dataset(n).kind == _BUNDLE]
    if bundle_targets and (force or not data_bundle.is_local()):
        data_bundle.fetch()
        downloaded += bundle_targets
    for n in targets:
        if dataset(n).kind == _HPA:
            already = reference_data.local_path(n).exists()
            reference_data.download(n, force=force)
            if force or not already:
                downloaded.append(n)
    return downloaded


def _size_bytes(p: Path | None) -> int:
    if p is None or not p.exists():
        return 0
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size


def _cohort_count(p: Path | None) -> int | None:
    """Per-cohort file count for a directory dataset (one file per cancer cohort);
    ``None`` for a single-file dataset. This is the cohort scale held *inside* an
    expression artifact — e.g. ``cancer-reference-expression-percentiles`` is one
    catalog entry but ~118 cancer cohorts."""
    if p is None or not p.is_dir():
        return None
    return sum(1 for f in p.iterdir() if f.is_file() and not f.name.startswith("_"))


def inventory() -> list[dict]:
    """The complete oncoref-domain data inventory — the full picture behind the
    fetchable :func:`datasets`. One row per dataset with ``name``, ``held``
    (``wheel`` / ``bundle`` / ``hpa`` / ``source`` / ``planned``), ``category``,
    ``available`` (present locally / shipped), and ``description``. Driven by
    :mod:`oncoref.data_manifest`; this is the current self-owned inventory.
    The separately versioned pirlygenes migration snapshot is historical audit
    evidence and does not expand when downstream adds a new dataset.
    """
    bundle_member = {p.removesuffix(".csv"): p for p in data_bundle.DOWNLOADABLE_PATHS}
    rows: list[dict] = []

    def _add(name, held, category, description, available, cohorts=None):
        rows.append(
            {
                "name": name,
                "held": held,
                "category": category,
                "available": available,
                "cohorts": cohorts,
                "description": description,
            }
        )

    wheel = {**data_manifest.WHEEL, **data_manifest.CANCERDATA_ORIGINATED}
    for name, (cat, desc) in sorted(wheel.items()):
        _add(name, "wheel", cat, desc, True)  # ships in the wheel — always present
    for name, (cat, desc) in sorted(data_manifest.BUNDLE.items()):
        member = data_bundle.find(bundle_member[name])
        _add(name, "bundle", cat, desc, member is not None, _cohort_count(member))
    for name, (cat, desc) in sorted(data_manifest.HPA.items()):
        _add(name, "hpa", cat, desc, reference_data.local_path(name).exists())
    for name, (cat, desc) in sorted(data_manifest.SOURCE.items()):
        cohorts = source_matrices.available_cohorts()
        available = any(source_matrices.is_cached(c) for c in cohorts)
        _add(name, "source", cat, desc, available, len(cohorts))  # per-cohort fetchable
    for name, (cat, desc) in sorted(data_manifest.PLANNED.items()):
        _add(name, "planned", cat, desc, False)  # oncoref-domain, still to port
    return rows


def status(name: str | None = None) -> list[dict]:
    """Uniform status rows over the catalog: ``name``, ``kind``, ``present``,
    ``path``, ``size_bytes``, ``cohorts`` (per-cohort file count for directory
    datasets, else ``None``), ``description``. One dataset if ``name`` given."""
    if name == "all":
        name = None
    if name is not None and name.startswith(_PER_SAMPLE):
        return [_per_sample_status_row(name)]
    if name == "source":
        return [
            _per_sample_status_row(f"{_PER_SAMPLE}{code}")
            for code in source_matrices.available_cohorts()
        ]
    if name in (_BUNDLE, _HPA):
        names = [d.name for d in datasets() if d.kind == name]
    else:
        names = [dataset(name).name] if name is not None else [d.name for d in datasets()]
    rows = []
    for n in names:
        d = dataset(n)
        p = path(n)
        rows.append(
            {
                "name": n,
                "kind": d.kind,
                "present": p is not None,
                "path": str(p) if p is not None else None,
                "size_bytes": _size_bytes(p),
                "cohorts": _cohort_count(p),
                "description": d.description,
            }
        )
    return rows
