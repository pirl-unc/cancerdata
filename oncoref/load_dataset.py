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

"""Loader for bundled + downloaded data files.

Two data roots are checked in order:

  1. ``_BUNDLED_DATA_DIR`` — files shipped in the wheel (the small curated
     tables) AND files present in a git checkout's ``oncoref/data/``.
  2. the cache populated by :mod:`oncoref.data_bundle` (the large per-cohort
     expression summaries fetched from the GitHub Release).

Any file present in (1) wins over (2). When a callable requests one of the
:data:`oncoref.data_bundle.DOWNLOADABLE_PATHS` items and it's missing from
both, :func:`oncoref.data_bundle.ensure_local` triggers a one-time download.
"""

from __future__ import annotations

import contextlib
import warnings
from pathlib import Path

import pandas as pd

from . import data_bundle

_BUNDLED_DATA_DIR = Path(__file__).parent / "data"
_DATASET_PATHS = None
_CACHED_DATAFRAMES: dict = {}

# Bump this when the representation written to the concatenated-shard parquet
# cache changes. The source-file signature alone cannot invalidate an older
# parquet file whose dtypes no longer match the owning cache contract.
_SHARD_CACHE_SCHEMA_VERSION = 2

# These columns repeat a small set of provenance values across millions of gene
# rows. Keeping them as Python objects dominates the resident size of the
# cancer-reference-expression frame. Categoricals preserve the values and
# comparisons while storing one compact integer code per row.
_CATEGORICAL_COLUMNS_BY_DATASET = {
    "cancer-reference-expression": (
        "cancer_code",
        "source_cohort",
        "source_project",
        "source_version",
        "processing_pipeline",
        "notes",
        "tumor_origin",
    ),
    "tcga-deconvolved-expression": ("Ensembl_Gene_ID", "symbol", "cancer_code"),
    "subtype-deconvolved-expression": (
        "Ensembl_Gene_ID",
        "symbol",
        "cancer_code",
        "subtype",
        "source_cohort",
    ),
    "tumor-reference-expression-provenance": (
        "artifact",
        "cancer_code",
        "subtype",
        "source_cohort",
        "derivation_method",
        "derivation_status",
        "source_scale",
        "processing_pipeline",
        "source_artifact",
        "source_artifact_commit",
        "source_matrix_version",
        "sample_qc_policy",
        "sample_qc_artifact",
        "notes",
    ),
}

# NCBI uses literal aliases such as "NA" and "NaN". Preserve every nonempty
# string in this table while continuing to represent empty CSV cells as missing.
_CSV_READ_OPTIONS_BY_DATASET = {
    "ncbi-symbol-synonyms": {
        "dtype": "string",
        "keep_default_na": False,
        "na_values": [""],
    },
}

# Back-compat alias.
_DATA_DIR = _BUNDLED_DATA_DIR

#: Clear callbacks for caches DERIVED from get_data results (e.g. processed
#: lookups in incidence / cta). Each cache-holder registers its ``cache_clear`` via
#: :func:`_register_derived_cache`; :func:`_clear_cache` drives them so swapping a
#: bundled fixture invalidates the derived views too — no module needs to know
#: about another's caches.
_DERIVED_CACHE_CLEARERS: list = []


def _register_derived_cache(clear_fn) -> None:
    """Register a ``cache_clear``-style callback to run on :func:`_clear_cache`."""
    _DERIVED_CACHE_CLEARERS.append(clear_fn)


def _clear_cache() -> None:
    """Drop cached frames + dataset-path map + every derived cache. Test hook for
    swapping fixtures."""
    _CACHED_DATAFRAMES.clear()
    _invalidate_dataset_paths()
    for clear_fn in _DERIVED_CACHE_CLEARERS:
        clear_fn()


def _data_roots() -> list[Path]:
    """Roots checked when resolving a data file, in priority order."""
    return [_BUNDLED_DATA_DIR, data_bundle.cache_dir()]


def _ensure_downloadable(name: str) -> None:
    """Fetch the bundle if ``name`` maps to a downloadable item missing from
    BOTH the bundled checkout AND the cache. No-op when present in either."""
    downloadable_path = data_bundle._canonical_downloadable_path(name)
    if downloadable_path is None or data_bundle.find_local_item(downloadable_path) is not None:
        return
    data_bundle.ensure_local()


def _shard_directories() -> list[Path]:
    """Subdirectories holding sharded CSV datasets, gathered from both roots.

    A shard directory ``<root>/<name>/`` containing one or more ``*.csv.gz``
    files acts as a single logical dataset addressable as ``<name>`` via
    :func:`get_data`; its shards are concatenated transparently.
    """
    seen: dict[str, Path] = {}
    for root in _data_roots():
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if any(child.glob("*.csv")) or any(child.glob("*.csv.gz")):
                seen.setdefault(child.name, child)
    return [seen[name] for name in sorted(seen)]


def _shard_paths(shard_dir: Path) -> list[Path]:
    return sorted(list(shard_dir.glob("*.csv")) + list(shard_dir.glob("*.csv.gz")))


def get_all_csv_paths() -> list:
    """Paths to every top-level CSV file across both data roots (bundled wins)."""
    seen: dict[str, Path] = {}
    for root in _data_roots():
        if not root.exists():
            continue
        for p in sorted(list(root.glob("*.csv")) + list(root.glob("*.csv.gz"))):
            seen.setdefault(p.name, p)
    return list(seen.values())


def _optimize_cached_dataframe(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply the owning cache's compact dtype policy for ``dataset_name``."""
    for column in _CATEGORICAL_COLUMNS_BY_DATASET.get(dataset_name, ()):
        if column in df.columns and not isinstance(df[column].dtype, pd.CategoricalDtype):
            df[column] = df[column].astype("category")
    return df


def _read_csv_for_cache(path: Path, dataset_name: str) -> pd.DataFrame:
    """Read one CSV with the owning dataset's lossless parsing contract."""
    options = _CSV_READ_OPTIONS_BY_DATASET.get(dataset_name, {})
    return pd.read_csv(str(path), low_memory=False, **options)


def _dataset_key(name: str) -> str:
    """Return one case-insensitive cache key for logical and physical CSV names."""
    return str(name).lower().removesuffix(".gz").removesuffix(".csv")


def _read_shards_for_cache(shard_dir: Path, paths: list[Path]) -> list[pd.DataFrame]:
    """Read shards while compacting each one before the next is retained."""
    frames = []
    for path in paths:
        frame = _read_csv_for_cache(path, shard_dir.name)
        frames.append(_optimize_cached_dataframe(shard_dir.name, frame))
    return frames


def _load_shard_directory(shard_dir: Path) -> pd.DataFrame:
    """Concatenate every ``*.csv[.gz]`` shard in a sharded dataset directory.

    Keeps a best-effort parquet cache of the concatenated frame in
    ``~/.cache/oncoref/shard_cache/``, keyed on a signature of the shard
    files (count + total size + newest mtime), auto-invalidating on change.
    """
    paths = _shard_paths(shard_dir)
    if not paths:
        raise FileNotFoundError(f"no CSV shards found under {shard_dir}")
    sig = repr(
        (
            _SHARD_CACHE_SCHEMA_VERSION,
            len(paths),
            sum(p.stat().st_size for p in paths),
            max(p.stat().st_mtime_ns for p in paths),
        )
    )
    cache_dir = Path.home() / ".cache" / "oncoref" / "shard_cache"
    cache_file = cache_dir / f"{shard_dir.name}.parquet"
    sig_file = cache_dir / f"{shard_dir.name}.sig"
    try:
        if cache_file.exists() and sig_file.exists() and sig_file.read_text() == sig:
            cached = pd.read_parquet(cache_file)
            return _optimize_cached_dataframe(shard_dir.name, cached)
    except Exception as e:
        # Corrupt/unreadable cache: self-heal by removing it (so it doesn't fail
        # every run) and surface a warning, then rebuild from the authoritative
        # CSVs below.
        warnings.warn(
            f"oncoref: rebuilding unreadable shard cache {cache_file.name}: {e}",
            stacklevel=2,
        )
        for stale in (cache_file, sig_file):
            with contextlib.suppress(OSError):
                stale.unlink()
    df = pd.concat(_read_shards_for_cache(shard_dir, paths), ignore_index=True)
    _optimize_cached_dataframe(shard_dir.name, df)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
        sig_file.write_text(sig)
    except Exception as e:
        # Caching is best-effort — a write failure (disk full, read-only FS)
        # must never fail the load, but make it visible.
        warnings.warn(
            f"oncoref: could not write shard cache {cache_file.name}: {e}",
            stacklevel=2,
        )
    return df


def _load_resolved_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    """Load one resolved file or shard directory into its optimized owning form."""
    if path.is_dir():
        return _load_shard_directory(path)
    frame = _read_csv_for_cache(path, dataset_name)
    return _optimize_cached_dataframe(dataset_name, frame)


def _invalidate_dataset_paths() -> None:
    global _DATASET_PATHS
    _DATASET_PATHS = None


def _dataset_paths():
    """Map accepted dataset names to their on-disk CSV path or shard dir."""
    global _DATASET_PATHS
    if _DATASET_PATHS is not None:
        return _DATASET_PATHS

    paths: dict[str, Path] = {}
    for csv_path in get_all_csv_paths():
        paths[_dataset_key(csv_path.name)] = csv_path
    for shard_dir in _shard_directories():
        paths[_dataset_key(shard_dir.name)] = shard_dir
    _DATASET_PATHS = paths
    return paths


def get_data(name, _dataframes_dict=None, *, copy=True):
    """Load a packaged dataset as a DataFrame.

    By default returns a defensive ``.copy()`` so callers that mutate in place
    can't corrupt the shared cache. Pass ``copy=False`` for read-only callers
    that slice/copy before mutating.
    """
    dataset_name = _dataset_key(name)

    if _dataframes_dict is not None:
        fixtures_by_name = {_dataset_key(key): frame for key, frame in _dataframes_dict.items()}
        if dataset_name in fixtures_by_name:
            frame = fixtures_by_name[dataset_name]
            return frame.copy() if copy else frame
        raise ValueError(f"Dataset {name} not found")

    # Trigger download for downloadable items before resolving paths.
    _ensure_downloadable(name)
    paths = _dataset_paths()

    if dataset_name not in paths and data_bundle.is_downloadable(name):
        data_bundle.ensure_local()
        _invalidate_dataset_paths()
        paths = _dataset_paths()

    resolved = paths.get(dataset_name)
    if resolved is None:
        raise ValueError(f"Dataset {name} not found")
    if dataset_name not in _CACHED_DATAFRAMES:
        _CACHED_DATAFRAMES[dataset_name] = _load_resolved_dataset(resolved, dataset_name)
    cached = _CACHED_DATAFRAMES[dataset_name]
    return cached.copy() if copy else cached
