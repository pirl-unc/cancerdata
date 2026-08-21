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

"""Lazy download of the large per-cohort expression bundle from a GitHub Release.

The wheel ships small curated tables directly (cancer-type registry, TMB,
incidence/mortality — ~60 KB). The much larger per-cohort expression summaries
and percentile/vector artifacts download on first access from the version-pinned
GitHub Release.

Cache layout (version-pinned so upgrades trigger a re-fetch):

  ~/.cache/oncoref/bundled_data/v<DATA_VERSION>/
    cancer-reference-expression/...
    cancer-reference-expression-percentiles/...
    cancer-reference-expression-percentiles-proteoform-cta/...
    pan-cancer-expression.csv
    hpa-cell-type-expression.csv
    tcga-deconvolved-expression.csv.gz
    subtype-deconvolved-expression.csv.gz
    tumor-reference-expression-provenance.csv
    source-matrix-sample-qc.csv
    expression-artifact-build-metadata.*

oncoref now owns the bundle: new downloads land under ``~/.cache/oncoref``
and use the checksum-verified ``pirl-unc/oncoref`` release for the active
``DATA_VERSION``. To avoid a forced re-download during the migration, an
already-populated legacy ``~/.cache/pirlygenes`` cache for the current version
is reused as-is. The ``PIRLYGENES_BUNDLED_DATA`` env var is still honored;
``CANCERDATA_BUNDLED_DATA`` takes precedence when set.

Public API:
  cache_dir()      → version-pinned cache Path
  bundle_is_local() → bool: every downloadable path present in the cache?
  item_is_local()  → bool: one item present in the package or cache?
  fetch()          → download + extract from the GitHub Release
  ensure_local()   → fetch if missing; safe to call on every access
  find(path)       → cached path or None
  find_local_item(path) → packaged or cached path, without fetching
  status()         → dict summarizing local state
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from filelock import FileLock

from .version import DATA_VERSION, SOURCE_MATRIX_VERSION, __version__


def _release_url(repo: str, filename: str) -> str:
    return f"https://github.com/{repo}/releases/download/v{DATA_VERSION}/{filename}"


# oncoref owns the bundle: its own release is the checksum-verified source.
# The historical pirlygenes release metadata remains here for explicit migration
# audits, but the primary oncoref source must be complete for normal fetches.
GITHUB_REPO = "pirl-unc/oncoref"
TARBALL_FILENAME = f"oncoref-data-v{DATA_VERSION}.tar.gz"
RELEASE_URL = _release_url(GITHUB_REPO, TARBALL_FILENAME)
MANIFEST_FILENAME = f"oncoref-data-v{DATA_VERSION}.manifest.json"
CHECKSUM_FILENAME = f"{TARBALL_FILENAME}.sha256"
RELEASE_MANIFEST_URL = _release_url(GITHUB_REPO, MANIFEST_FILENAME)
RELEASE_CHECKSUM_URL = _release_url(GITHUB_REPO, CHECKSUM_FILENAME)

FALLBACK_GITHUB_REPO = "pirl-unc/pirlygenes"
FALLBACK_TARBALL_FILENAME = f"pirlygenes-data-v{DATA_VERSION}.tar.gz"
FALLBACK_RELEASE_URL = _release_url(FALLBACK_GITHUB_REPO, FALLBACK_TARBALL_FILENAME)
FALLBACK_MANIFEST_FILENAME = f"pirlygenes-data-v{DATA_VERSION}.manifest.json"
FALLBACK_CHECKSUM_FILENAME = f"{FALLBACK_TARBALL_FILENAME}.sha256"
FALLBACK_RELEASE_MANIFEST_URL = _release_url(FALLBACK_GITHUB_REPO, FALLBACK_MANIFEST_FILENAME)
FALLBACK_RELEASE_CHECKSUM_URL = _release_url(FALLBACK_GITHUB_REPO, FALLBACK_CHECKSUM_FILENAME)

RELEASE_SOURCES: tuple[dict, ...] = (
    {
        "name": "oncoref",
        "repo": GITHUB_REPO,
        "url": RELEASE_URL,
        "tarball_filename": TARBALL_FILENAME,
        "manifest_url": RELEASE_MANIFEST_URL,
        "checksum_url": RELEASE_CHECKSUM_URL,
        "require_integrity": True,
    },
    {
        "name": "pirlygenes",
        "repo": FALLBACK_GITHUB_REPO,
        "url": FALLBACK_RELEASE_URL,
        "tarball_filename": FALLBACK_TARBALL_FILENAME,
        "manifest_url": FALLBACK_RELEASE_MANIFEST_URL,
        "checksum_url": FALLBACK_RELEASE_CHECKSUM_URL,
        "require_integrity": False,
    },
)

#: Release URLs tried in order until one downloads.
RELEASE_URLS: tuple[str, ...] = tuple(source["url"] for source in RELEASE_SOURCES)

CACHE_COMPLETE_FILENAME = ".oncoref-bundle-complete.json"
CACHE_MANIFEST_VERSION = 1
BUNDLE_MANIFEST_VERSION = 2
BUNDLE_CONTRACT_VERSION = 3


class BundleIntegrityError(RuntimeError):
    """Raised when a release manifest/checksum is missing or does not match."""


#: Env var that overrides the cache (points at the version-pinned dir).
CACHE_DIR_ENV_VAR = "CANCERDATA_BUNDLED_DATA"
#: Back-compat env var honored when this package's own override is unset.
LEGACY_CACHE_DIR_ENV_VAR = "PIRLYGENES_BUNDLED_DATA"

#: Default cache parents. New downloads go under the oncoref root; an existing
#: pirlygenes cache for the current version is reused to avoid a re-download.
_DEFAULT_CACHE_PARENT = Path.home() / ".cache" / "oncoref" / "bundled_data"
_LEGACY_CACHE_PARENT = Path.home() / ".cache" / "pirlygenes" / "bundled_data"
_PACKAGE_DATA_DIR = Path(__file__).parent / "data"

# Names that live in the downloadable tarball (relative to the cache root) and
# are NOT bundled in the wheel. load_dataset checks here after the wheel data dir.
DOWNLOADABLE_PATHS: tuple[str, ...] = (
    "cancer-reference-expression",  # directory of per-source shards
    "cancer-reference-expression-representatives",  # per-cohort medoid parquets
    "cancer-reference-expression-percentiles",  # per-gene percentile vectors
    "cancer-reference-expression-within-sample-top5",  # within-sample high-rank prevalence
    "cancer-reference-expression-percentiles-proteoform-cta",  # CTA proteoform percentiles
    "cancer-reference-expression-within-sample-top5-proteoform-cta",  # CTA proteoform prevalence
    "source-matrix-sample-qc.csv",  # per-sample QC statuses used by rebuilt artifacts
    "expression-artifact-build-metadata.csv",  # per-cohort build/QC provenance
    "expression-artifact-build-metadata.json",  # bundle-level build/QC provenance
    "pan-cancer-expression.csv",
    "hpa-cell-type-expression.csv",
    "tcga-deconvolved-expression.csv.gz",
    "subtype-deconvolved-expression.csv.gz",
    "tumor-reference-expression-provenance.csv",
)


def _cache_override() -> str | None:
    return os.environ.get(CACHE_DIR_ENV_VAR) or os.environ.get(LEGACY_CACHE_DIR_ENV_VAR)


def cache_root() -> Path:
    """Parent of all version-pinned cache dirs (``v<version>/`` lives inside).

    Defaults to the oncoref cache root, but if *this* version was already
    downloaded under the legacy pirlygenes root (and not yet under the new one),
    that legacy root is returned so the migration doesn't force a re-download.
    """
    override = _cache_override()
    if override:
        # Override points at the version-pinned dir; its parent is the root.
        return Path(override).expanduser().parent
    version_dir = f"v{DATA_VERSION}"
    if (
        not (_DEFAULT_CACHE_PARENT / version_dir).exists()
        and (_LEGACY_CACHE_PARENT / version_dir).exists()
    ):
        return _LEGACY_CACHE_PARENT
    return _DEFAULT_CACHE_PARENT


def cache_dir() -> Path:
    """Where the downloaded bundle lives on disk for this version."""
    override = _cache_override()
    if override:
        return Path(override).expanduser()
    return cache_root() / f"v{DATA_VERSION}"


def _cache_lock(root: Path) -> FileLock:
    """Cross-process lock for writes to one version-pinned bundle cache."""
    root.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(root.parent / f".{root.name}.oncoref.lock"))


def _path_complete(path: Path) -> bool:
    """A downloadable entry counts as present only if it actually holds data.

    A bare file must exist; a directory must contain at least one file. An
    interrupted extract that created the shard directories but no shards (or a
    legacy/empty cache dir) would otherwise read as "local" forever and never
    re-fetch — later reads then fail with confusing empties (issue #21)."""
    if not path.exists():
        return False
    if path.is_dir():
        return any(f.is_file() for f in path.rglob("*"))
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _entry_inventory(path: Path) -> dict:
    file_count = 0
    size_bytes = 0
    if path.exists():
        if path.is_dir():
            files = [f for f in path.rglob("*") if f.is_file()]
            file_count = len(files)
            size_bytes = sum((f.stat().st_size for f in files), start=0)
        else:
            file_count = 1
            size_bytes = path.stat().st_size
    return {
        "present": path.exists(),
        "complete": _path_complete(path),
        "file_count": file_count,
        "size_bytes": size_bytes,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_url_text(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def _parse_checksum_text(text: str, *, filename: str) -> dict:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    parts = first.split()
    if not parts:
        raise BundleIntegrityError(f"empty checksum file for {filename}")
    sha256 = parts[0].strip()
    if len(sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in sha256):
        raise BundleIntegrityError(f"invalid sha256 checksum for {filename}: {sha256!r}")
    recorded_name = parts[-1].lstrip("*") if len(parts) > 1 else filename
    if Path(recorded_name).name != filename:
        raise BundleIntegrityError(f"checksum file names {recorded_name!r}, expected {filename!r}")
    return {
        "manifest_version": BUNDLE_MANIFEST_VERSION,
        "data_version": DATA_VERSION,
        "tarball": {
            "filename": filename,
            "sha256": sha256.lower(),
        },
    }


def _valid_sha256(value: Any) -> str | None:
    sha256 = str(value or "").lower()
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        return None
    return sha256


def _bundle_relative_path(value: Any, *, manifest_url: str) -> str:
    relative = str(value or "")
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise BundleIntegrityError(f"{manifest_url} contains unsafe bundle path {relative!r}")
    if not any(relative == root or relative.startswith(f"{root}/") for root in DOWNLOADABLE_PATHS):
        raise BundleIntegrityError(
            f"{manifest_url} overlay path is outside the bundle contract: {relative!r}"
        )
    return relative


def _normalize_base_bundle(base: Any, *, manifest_url: str) -> dict:
    if not isinstance(base, dict):
        raise BundleIntegrityError(f"{manifest_url} overlay lacks a base_bundle object")
    data_version = str(base.get("data_version") or "")
    version_parts = data_version.split(".")
    if len(version_parts) != 3 or not all(part.isdigit() for part in version_parts):
        raise BundleIntegrityError(
            f"{manifest_url} base_bundle has invalid data_version {data_version!r}"
        )
    if data_version == DATA_VERSION:
        raise BundleIntegrityError(f"{manifest_url} overlay cannot use itself as its base")
    repo = str(base.get("repo") or GITHUB_REPO)
    if repo != GITHUB_REPO:
        raise BundleIntegrityError(
            f"{manifest_url} overlay base repo {repo!r} is not the oncoref authority"
        )
    tarball = base.get("tarball")
    if not isinstance(tarball, dict):
        raise BundleIntegrityError(f"{manifest_url} base_bundle lacks tarball metadata")
    filename = str(tarball.get("filename") or "")
    expected_filename = f"oncoref-data-v{data_version}.tar.gz"
    if filename != expected_filename:
        raise BundleIntegrityError(
            f"{manifest_url} base tarball is {filename!r}, expected {expected_filename!r}"
        )
    sha256 = _valid_sha256(tarball.get("sha256"))
    if sha256 is None:
        raise BundleIntegrityError(f"{manifest_url} base_bundle lacks a valid tarball sha256")
    try:
        size_bytes = int(tarball.get("bytes"))
    except (TypeError, ValueError) as e:
        raise BundleIntegrityError(f"{manifest_url} base_bundle lacks a valid byte size") from e
    if size_bytes <= 0:
        raise BundleIntegrityError(f"{manifest_url} base_bundle byte size must be positive")
    paths = tarball.get("downloadable_paths")
    if paths is not None and tuple(paths) != DOWNLOADABLE_PATHS:
        raise BundleIntegrityError(
            f"{manifest_url} base downloadable_paths do not match this oncoref build"
        )
    return {
        "data_version": data_version,
        "repo": repo,
        "tarball": {
            "filename": filename,
            "url": (f"https://github.com/{repo}/releases/download/v{data_version}/{filename}"),
            "bytes": size_bytes,
            "sha256": sha256,
            "downloadable_paths": list(DOWNLOADABLE_PATHS),
        },
    }


def _normalize_overlay(overlay: Any, *, manifest_url: str) -> dict:
    if not isinstance(overlay, dict):
        raise BundleIntegrityError(f"{manifest_url} overlay layout lacks overlay metadata")
    raw_paths = overlay.get("paths")
    raw_deleted_paths = overlay.get("deleted_paths", [])
    if not isinstance(raw_paths, list) or not isinstance(raw_deleted_paths, list):
        raise BundleIntegrityError(f"{manifest_url} overlay paths must be arrays")
    paths = [_bundle_relative_path(p, manifest_url=manifest_url) for p in raw_paths]
    deleted_paths = [_bundle_relative_path(p, manifest_url=manifest_url) for p in raw_deleted_paths]
    if len(paths) != len(set(paths)) or len(deleted_paths) != len(set(deleted_paths)):
        raise BundleIntegrityError(f"{manifest_url} overlay paths must be unique")
    if set(paths) & set(deleted_paths):
        raise BundleIntegrityError(f"{manifest_url} cannot replace and delete the same path")
    if overlay.get("file_count", len(paths)) != len(paths):
        raise BundleIntegrityError(f"{manifest_url} overlay file_count does not match its paths")
    try:
        size_bytes = int(overlay.get("size_bytes"))
    except (TypeError, ValueError) as e:
        raise BundleIntegrityError(f"{manifest_url} overlay lacks a valid byte size") from e
    if size_bytes < 0:
        raise BundleIntegrityError(f"{manifest_url} overlay byte size cannot be negative")
    return {
        "paths": paths,
        "deleted_paths": deleted_paths,
        "file_count": len(paths),
        "size_bytes": size_bytes,
    }


def _validate_release_manifest(manifest: dict, source: dict, *, manifest_url: str) -> dict:
    if manifest.get("data_version") != DATA_VERSION:
        raise BundleIntegrityError(
            f"{manifest_url} is for data_version {manifest.get('data_version')!r}, "
            f"expected {DATA_VERSION!r}"
        )
    if manifest.get("manifest_version") not in (None, 1, BUNDLE_MANIFEST_VERSION):
        raise BundleIntegrityError(
            f"{manifest_url} uses unsupported manifest_version {manifest.get('manifest_version')!r}"
        )
    tarball = dict(manifest.get("tarball") or {})
    filename = tarball.get("filename") or source["tarball_filename"]
    if filename != source["tarball_filename"]:
        raise BundleIntegrityError(
            f"{manifest_url} describes tarball {filename!r}, "
            f"expected {source['tarball_filename']!r}"
        )
    sha256 = _valid_sha256(tarball.get("sha256"))
    if sha256 is None:
        raise BundleIntegrityError(f"{manifest_url} lacks a valid tarball sha256")
    paths = tarball.get("downloadable_paths") or manifest.get("downloadable_paths")
    if paths is not None and tuple(paths) != DOWNLOADABLE_PATHS:
        raise BundleIntegrityError(
            f"{manifest_url} downloadable_paths do not match this oncoref build"
        )
    manifest_source_matrix_version = manifest.get("source_matrix_version")
    if (
        manifest_source_matrix_version is not None
        and str(manifest_source_matrix_version) != SOURCE_MATRIX_VERSION
    ):
        raise BundleIntegrityError(
            f"{manifest_url} is for source_matrix_version {manifest_source_matrix_version!r}, "
            f"expected {SOURCE_MATRIX_VERSION!r}"
        )
    inventory = manifest.get("inventory")
    if inventory is not None:
        if not isinstance(inventory, dict):
            raise BundleIntegrityError(f"{manifest_url} inventory must be an object")
        missing_inventory = [
            path for path in DOWNLOADABLE_PATHS if not isinstance(inventory.get(path), dict)
        ]
        if missing_inventory:
            raise BundleIntegrityError(
                f"{manifest_url} inventory lacks required bundle paths: "
                + ", ".join(missing_inventory)
            )
    normalized = {
        "manifest_version": manifest.get("manifest_version", BUNDLE_MANIFEST_VERSION),
        "data_version": DATA_VERSION,
        "source": source["name"],
        "repo": source["repo"],
        "manifest_url": manifest_url,
        "tarball": {
            "filename": filename,
            "url": source["url"],
            "sha256": sha256,
            "bytes": tarball.get("bytes"),
            "downloadable_paths": list(DOWNLOADABLE_PATHS),
        },
    }
    if isinstance(inventory, dict):
        normalized["inventory"] = {
            path: inventory.get(path)
            for path in DOWNLOADABLE_PATHS
            if isinstance(inventory.get(path), dict)
        }
    layout = manifest.get("bundle_layout", "full")
    if layout not in {"full", "overlay"}:
        raise BundleIntegrityError(f"{manifest_url} has unsupported bundle_layout {layout!r}")
    normalized["bundle_layout"] = layout
    if layout == "overlay":
        if manifest.get("manifest_version") != BUNDLE_MANIFEST_VERSION:
            raise BundleIntegrityError(
                f"{manifest_url} overlay requires manifest_version {BUNDLE_MANIFEST_VERSION}"
            )
        normalized["base_bundle"] = _normalize_base_bundle(
            manifest.get("base_bundle"), manifest_url=manifest_url
        )
        normalized["overlay"] = _normalize_overlay(
            manifest.get("overlay"), manifest_url=manifest_url
        )
    for key in (
        "builder",
        "builder_commit",
        "created_at",
        "package_version",
        "source_matrix_version",
        "sample_qc_policy",
        "sample_qc_policy_version",
        "source_matrix_sample_qc",
        "artifact_build_metadata",
    ):
        if key in manifest:
            normalized[key] = manifest[key]
    return normalized


def _fetch_release_manifest(source: dict) -> dict | None:
    """Fetch and validate the release manifest/checksum for a bundle source.

    The oncoref-owned source is strict: a missing manifest/checksum is a release
    contract violation. The pirlygenes source is legacy migration fallback; it is
    used only when oncoref's tarball is absent and remains marked unverified if it
    has no same-release checksum.
    """
    manifest_url = source["manifest_url"]
    try:
        manifest = json.loads(_read_url_text(manifest_url))
        return _validate_release_manifest(manifest, source, manifest_url=manifest_url)
    except urllib.error.HTTPError as manifest_error:
        if manifest_error.code != 404:
            raise
    except json.JSONDecodeError as e:
        raise BundleIntegrityError(f"{manifest_url} is not valid JSON: {e}") from e

    checksum_url = source["checksum_url"]
    try:
        manifest = _parse_checksum_text(
            _read_url_text(checksum_url),
            filename=source["tarball_filename"],
        )
        return _validate_release_manifest(manifest, source, manifest_url=checksum_url)
    except urllib.error.HTTPError as checksum_error:
        if checksum_error.code != 404:
            raise
        if source["require_integrity"]:
            raise BundleIntegrityError(
                f"missing release manifest/checksum for {source['url']} "
                f"(tried {manifest_url} and {checksum_url})"
            ) from checksum_error
        return None


def _bundle_inventory(root: Path) -> dict[str, dict]:
    return {p: _entry_inventory(root / p) for p in DOWNLOADABLE_PATHS}


def _incomplete_bundle_paths(root: Path) -> list[str]:
    return [p for p in DOWNLOADABLE_PATHS if not _path_complete(root / p)]


def _completion_marker_path(root: Path) -> Path:
    return root / CACHE_COMPLETE_FILENAME


def _read_completion_marker(root: Path) -> dict | None:
    marker = _completion_marker_path(root)
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _completion_marker_valid(root: Path) -> bool:
    marker = _read_completion_marker(root)
    release_manifest = marker.get("release_manifest") if marker else None
    tarball = release_manifest.get("tarball") if isinstance(release_manifest, dict) else None
    return bool(
        marker
        and marker.get("manifest_version") == CACHE_MANIFEST_VERSION
        and marker.get("data_version") == DATA_VERSION
        and tuple(marker.get("downloadable_paths", ())) == DOWNLOADABLE_PATHS
        and isinstance(tarball, dict)
        and tarball.get("sha256")
        and not _incomplete_bundle_paths(root)
    )


def _base_cache_valid(root: Path, base_bundle: dict) -> bool:
    marker = _read_completion_marker(root)
    release_manifest = marker.get("release_manifest") if marker else None
    tarball = release_manifest.get("tarball") if isinstance(release_manifest, dict) else None
    expected = base_bundle["tarball"]
    return bool(
        marker
        and marker.get("manifest_version") == CACHE_MANIFEST_VERSION
        and marker.get("data_version") == base_bundle["data_version"]
        and tuple(marker.get("downloadable_paths", ())) == DOWNLOADABLE_PATHS
        and isinstance(tarball, dict)
        and tarball.get("sha256") == expected["sha256"]
        and not _incomplete_bundle_paths(root)
    )


def _write_completion_marker(
    root: Path,
    *,
    source_url: str,
    release_manifest: dict | None,
    data_version: str = DATA_VERSION,
) -> None:
    marker = _completion_marker_path(root)
    payload = {
        "manifest_version": CACHE_MANIFEST_VERSION,
        "data_version": data_version,
        "source_url": source_url,
        "release_manifest": release_manifest,
        "verified_sha256": bool(
            release_manifest and release_manifest.get("tarball", {}).get("sha256")
        ),
        "downloadable_paths": list(DOWNLOADABLE_PATHS),
        "inventory": _bundle_inventory(root),
    }
    tmp = marker.with_suffix(marker.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, marker)


def bundle_is_local() -> bool:
    """Whether every downloadable item is non-empty in the active cache."""
    root = cache_dir()
    return all(_path_complete(root / p) for p in DOWNLOADABLE_PATHS)


def is_local() -> bool:
    """Backward-compatible alias for :func:`bundle_is_local`."""
    return bundle_is_local()


def find(relative_path: str) -> Path | None:
    """Resolve a non-empty item in the active bundle cache, or return ``None``."""
    candidate = cache_dir() / relative_path
    return candidate if _path_complete(candidate) else None


def find_local_item(relative_path: str) -> Path | None:
    """Resolve one non-empty data item without downloading anything.

    The package data directory is checked before the active bundle cache, matching
    normal read precedence. This deliberately answers a narrower question than
    :func:`bundle_is_local`: whether this exact item is locally present and non-empty,
    including an artifact checkout or a partial developer cache. It does not perform
    checksum or schema validation.
    """
    path = Path(relative_path)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("relative_path must stay within a data root")
    for root in (_PACKAGE_DATA_DIR, cache_dir()):
        candidate = root / path
        if _path_complete(candidate):
            return candidate
    return None


def local_item_paths(relative_path: str) -> tuple[Path, ...]:
    """Return every non-empty local copy of one data item, without downloading.

    Package/artifact checkouts and the active bundle cache can each contain a
    partial shard directory.  Callers that enumerate shards must inspect both
    roots instead of stopping at the first non-empty directory.  Paths retain
    normal read precedence (package data first, then cache) and are de-duplicated
    when both roots resolve to the same location.
    """
    path = Path(relative_path)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("relative_path must stay within a data root")

    found: list[Path] = []
    seen: set[Path] = set()
    for root in (_PACKAGE_DATA_DIR, cache_dir()):
        candidate = root / path
        resolved = candidate.resolve()
        if _path_complete(candidate) and resolved not in seen:
            found.append(candidate)
            seen.add(resolved)
    return tuple(found)


def item_is_local(relative_path: str) -> bool:
    """Whether one data item is non-empty in the package or active cache."""
    return find_local_item(relative_path) is not None


def bundle_contract() -> dict:
    """Stable downstream contract for the active expression data bundle.

    This is the single lightweight object a downstream package can inspect to
    know which package/data/source-matrix versions belong together, which release
    assets should exist, which cache override variables are honored, and which
    artifact paths make up a complete bundle.
    """
    sources = [
        {
            "name": source["name"],
            "repo": source["repo"],
            "tarball_filename": source["tarball_filename"],
            "release_url": source["url"],
            "manifest_url": source["manifest_url"],
            "checksum_url": source["checksum_url"],
            "require_integrity": bool(source["require_integrity"]),
        }
        for source in RELEASE_SOURCES
    ]
    return {
        "contract_version": BUNDLE_CONTRACT_VERSION,
        "package_version": __version__,
        "data_version": DATA_VERSION,
        "source_matrix_version": SOURCE_MATRIX_VERSION,
        "cache_dir_env_var": CACHE_DIR_ENV_VAR,
        "legacy_cache_dir_env_var": LEGACY_CACHE_DIR_ENV_VAR,
        "cache_complete_filename": CACHE_COMPLETE_FILENAME,
        "cache_manifest_version": CACHE_MANIFEST_VERSION,
        "bundle_manifest_version": BUNDLE_MANIFEST_VERSION,
        "downloadable_paths": list(DOWNLOADABLE_PATHS),
        "release_sources": sources,
        "primary_release_source": sources[0],
    }


def bundle_release_manifest(source: str = "oncoref") -> dict | None:
    """Fetch and validate the small release manifest/checksum for ``DATA_VERSION``.

    This is a metadata-only inspection helper for downstream packages that need to
    decide whether the active data version has a published, checksum-anchored bundle
    before downloading hundreds of MB. The returned manifest includes the tarball
    sha256 and any release-side artifact inventory / builder metadata present in
    the release asset. The transitional ``pirlygenes`` source can return ``None`` if
    no manifest/checksum exists because it is explicitly not the integrity authority.
    """
    source_record = next((s for s in RELEASE_SOURCES if s["name"] == source), None)
    if source_record is None:
        supported = ", ".join(s["name"] for s in RELEASE_SOURCES)
        raise ValueError(f"unknown bundle release source {source!r}; supported: {supported}")
    return _fetch_release_manifest(source_record)


def bundle_metadata(
    source: str = "oncoref",
    *,
    include_release_manifest: bool = True,
) -> dict:
    """Machine-readable bundle dependency state for downstream packages.

    This composes the static bundle contract, local cache status, and optionally
    the small validated release manifest for the active ``DATA_VERSION``. It never
    downloads or extracts the heavy tarball; callers that only need to decide
    whether a dependency is usable can inspect this first, then call
    :func:`ensure_local` or :func:`fetch` deliberately.
    """
    snap = status()
    release_manifest = bundle_release_manifest(source) if include_release_manifest else None
    return {
        "contract_version": BUNDLE_CONTRACT_VERSION,
        "package_version": __version__,
        "data_version": DATA_VERSION,
        "source_matrix_version": SOURCE_MATRIX_VERSION,
        "release_source": source,
        "contract": snap["contract"],
        "local_cache": {
            "cache_dir": snap["cache_dir"],
            "all_local": snap["all_local"],
            "completion_marker": snap["completion_marker"],
            "inventory": snap["items"],
        },
        "release_manifest": release_manifest,
    }


def _download_and_extract(
    url: str,
    root: Path,
    *,
    verbose: bool,
    release_manifest: dict | None = None,
    data_version: str = DATA_VERSION,
) -> None:
    """Download a tarball from ``url`` and extract it into ``root``.

    Extraction goes into a staging dir first and is promoted into ``root`` only
    after it completes, so a failed/corrupt download never leaves a half-populated
    cache that a later run would read as valid. Raises ``urllib.error.URLError``
    (unreachable, e.g. 404) or ``tarfile.TarError`` (corrupt/non-tar body) — the
    caller falls back to the next source on either.
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    try:
        with urllib.request.urlopen(url) as resp, tmp_path.open("wb") as h:
            shutil.copyfileobj(resp, h, length=1024 * 1024)
        if release_manifest:
            tarball = release_manifest["tarball"]
            expected_bytes = tarball.get("bytes")
            if expected_bytes is not None and tmp_path.stat().st_size != int(expected_bytes):
                raise BundleIntegrityError(
                    f"{url} size mismatch: got {tmp_path.stat().st_size} bytes, "
                    f"expected {expected_bytes}"
                )
            actual_sha256 = _sha256_file(tmp_path)
            if actual_sha256 != tarball["sha256"]:
                raise BundleIntegrityError(
                    f"{url} sha256 mismatch: got {actual_sha256}, expected {tarball['sha256']}"
                )
        if verbose:
            sys.stderr.write("oncoref: extracting...\n")
            sys.stderr.flush()
        with tarfile.open(tmp_path) as tf:
            # filter=data is Python 3.12+; fall back to the older API.
            try:
                tf.extractall(staging, filter="data")
            except TypeError:
                tf.extractall(staging)
        incomplete = _incomplete_bundle_paths(staging)
        if incomplete:
            raise tarfile.TarError(
                "bundle tarball is missing or has empty required paths: " + ", ".join(incomplete)
            )
        # Promote staged entries into the cache, replacing any prior copy.
        _completion_marker_path(root).unlink(missing_ok=True)
        for entry in staging.iterdir():
            dest = root / entry.name
            if dest.is_dir():
                shutil.rmtree(dest)
            elif dest.exists():
                dest.unlink()
            shutil.move(str(entry), str(dest))
        _write_completion_marker(
            root,
            source_url=url,
            release_manifest=release_manifest,
            data_version=data_version,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


def _link_or_copy(source: str, destination: str) -> str:
    """Hard-link unchanged cached artifacts, falling back across filesystems."""
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _overlay_file_members(archive: tarfile.TarFile) -> list[str]:
    files: list[str] = []
    for member in archive.getmembers():
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise BundleIntegrityError(f"overlay archive contains unsafe path {member.name!r}")
        if member.isfile():
            files.append(path.as_posix())
        elif not member.isdir():
            raise BundleIntegrityError(
                f"overlay archive contains unsupported member {member.name!r}"
            )
    return files


def _download_and_apply_overlay(
    url: str,
    root: Path,
    *,
    base_root: Path,
    verbose: bool,
    release_manifest: dict,
) -> None:
    """Compose a complete cache from a verified base and a small overlay archive."""
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    overlay_staging = Path(tempfile.mkdtemp(prefix=".overlay-", dir=root.parent))
    composition = Path(tempfile.mkdtemp(prefix=".composition-", dir=root.parent))
    try:
        with urllib.request.urlopen(url) as resp, tmp_path.open("wb") as handle:
            shutil.copyfileobj(resp, handle, length=1024 * 1024)
        tarball = release_manifest["tarball"]
        expected_bytes = tarball.get("bytes")
        if expected_bytes is not None and tmp_path.stat().st_size != int(expected_bytes):
            raise BundleIntegrityError(
                f"{url} size mismatch: got {tmp_path.stat().st_size} bytes, "
                f"expected {expected_bytes}"
            )
        actual_sha256 = _sha256_file(tmp_path)
        if actual_sha256 != tarball["sha256"]:
            raise BundleIntegrityError(
                f"{url} sha256 mismatch: got {actual_sha256}, expected {tarball['sha256']}"
            )
        if verbose:
            sys.stderr.write("oncoref: composing verified base bundle and overlay...\n")
            sys.stderr.flush()
        with tarfile.open(tmp_path) as archive:
            archived_paths = _overlay_file_members(archive)
            declared_paths = release_manifest["overlay"]["paths"]
            if archived_paths != declared_paths:
                raise BundleIntegrityError(
                    f"{url} archive paths do not match the release overlay manifest"
                )
            try:
                archive.extractall(overlay_staging, filter="data")
            except TypeError:
                archive.extractall(overlay_staging)

        for relative in DOWNLOADABLE_PATHS:
            source = base_root / relative
            destination = composition / relative
            if source.is_dir():
                shutil.copytree(source, destination, copy_function=_link_or_copy)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                _link_or_copy(str(source), str(destination))

        for relative in release_manifest["overlay"]["deleted_paths"]:
            destination = composition / relative
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink(missing_ok=True)
        for relative in release_manifest["overlay"]["paths"]:
            source = overlay_staging / relative
            if not source.is_file():
                raise BundleIntegrityError(f"{url} is missing overlay file {relative!r}")
            destination = composition / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.unlink(missing_ok=True)
            shutil.move(str(source), str(destination))

        incomplete = _incomplete_bundle_paths(composition)
        if incomplete:
            raise tarfile.TarError(
                "composed bundle is missing or has empty required paths: " + ", ".join(incomplete)
            )

        root.mkdir(parents=True, exist_ok=True)
        _completion_marker_path(root).unlink(missing_ok=True)
        for relative in DOWNLOADABLE_PATHS:
            destination = root / relative
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
            shutil.move(str(composition / relative), str(destination))
        _write_completion_marker(root, source_url=url, release_manifest=release_manifest)
    finally:
        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(overlay_staging, ignore_errors=True)
        shutil.rmtree(composition, ignore_errors=True)


def _base_cache_candidates(root: Path, data_version: str) -> tuple[Path, ...]:
    candidates = [
        root.parent / f"v{data_version}",
        _DEFAULT_CACHE_PARENT / f"v{data_version}",
        _LEGACY_CACHE_PARENT / f"v{data_version}",
    ]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique and candidate != root:
            unique.append(candidate)
    return tuple(unique)


def _ensure_overlay_base(root: Path, base_bundle: dict, *, verbose: bool) -> Path:
    for candidate in _base_cache_candidates(root, base_bundle["data_version"]):
        if _base_cache_valid(candidate, base_bundle):
            if verbose:
                sys.stderr.write(f"oncoref: reusing verified base bundle at {candidate}\n")
                sys.stderr.flush()
            return candidate

    destination = _base_cache_candidates(root, base_bundle["data_version"])[0]
    if verbose:
        sys.stderr.write(
            f"oncoref: verified base v{base_bundle['data_version']} is not cached; "
            "downloading it once\n"
        )
        sys.stderr.flush()
    base_manifest = {
        "manifest_version": 1,
        "data_version": base_bundle["data_version"],
        "source": "oncoref",
        "repo": base_bundle["repo"],
        "tarball": base_bundle["tarball"],
    }
    destination.mkdir(parents=True, exist_ok=True)
    with _cache_lock(destination):
        if not _base_cache_valid(destination, base_bundle):
            _download_and_extract(
                base_bundle["tarball"]["url"],
                destination,
                verbose=verbose,
                release_manifest=base_manifest,
                data_version=base_bundle["data_version"],
            )
    return destination


def _fetch_into_cache(root: Path, *, verbose: bool) -> Path:
    """Download and replace ``root`` while its cache lock is held.

    The oncoref-owned source must publish a same-release manifest or sha256 asset
    for the exact ``DATA_VERSION``; missing assets, failed downloads, and checksum
    failures stop the fetch instead of silently relying on a cross-project
    fallback.
    """
    root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for source in RELEASE_SOURCES:
        url = source["url"]
        if verbose:
            sys.stderr.write(
                f"oncoref: downloading data bundle for v{DATA_VERSION} "
                "(large, one-time)\n"
                f"  from {url}\n"
                f"  to   {root}\n"
            )
            sys.stderr.flush()
        try:
            release_manifest = _fetch_release_manifest(source)
            if release_manifest and release_manifest.get("bundle_layout") == "overlay":
                base_root = _ensure_overlay_base(
                    root,
                    release_manifest["base_bundle"],
                    verbose=verbose,
                )
                _download_and_apply_overlay(
                    url,
                    root,
                    base_root=base_root,
                    verbose=verbose,
                    release_manifest=release_manifest,
                )
            else:
                _download_and_extract(
                    url,
                    root,
                    verbose=verbose,
                    release_manifest=release_manifest,
                )
        except BundleIntegrityError as e:
            errors.append(f"{url}: {e}")
            if source["require_integrity"]:
                raise RuntimeError(
                    "oncoref: data bundle release integrity check failed:\n  " + "\n  ".join(errors)
                ) from e
            if verbose:
                sys.stderr.write(
                    f"oncoref: {url} integrity unavailable ({e}); trying next source\n"
                )
                sys.stderr.flush()
            continue
        except (urllib.error.URLError, tarfile.TarError) as e:
            # URLError covers HTTPError (404); TarError covers a corrupt body or
            # an HTML error page served with 200.
            errors.append(f"{url}: {e}")
            if source["require_integrity"]:
                raise RuntimeError(
                    "oncoref: data bundle release download failed for the "
                    "checksum-verified primary source:\n  " + "\n  ".join(errors)
                ) from e
            if verbose:
                sys.stderr.write(f"oncoref: {url} unavailable ({e}); trying next source\n")
                sys.stderr.flush()
            continue
        if verbose:
            sys.stderr.write(f"oncoref: data bundle ready at {root}\n")
            sys.stderr.flush()
        return root
    raise RuntimeError(
        "oncoref: could not download the data bundle from any release source:\n  "
        + "\n  ".join(errors)
    )


def fetch(*, verbose: bool = True) -> Path:
    """Download + extract the bundle for this version into the cache.

    Concurrent writers are serialized. This operation always overwrites the
    active version so callers can deliberately repair a corrupt cache.
    """
    root = cache_dir()
    with _cache_lock(root):
        return _fetch_into_cache(root, verbose=verbose)


def ensure_local(*, auto_fetch: bool = True, verbose: bool = True) -> Path:
    """Make sure the bundle is present locally; download if not.

    With ``auto_fetch=False``, raises ``FileNotFoundError`` instead of
    triggering a network call — for read-only inspection paths that shouldn't
    surprise users with a large download.
    """
    root = cache_dir()
    if bundle_is_local():
        return root
    if not auto_fetch:
        raise FileNotFoundError(
            f"oncoref data bundle not found at {root}. "
            "Run `oncoref data fetch bundle` to download it."
        )
    with _cache_lock(root):
        # Another process may have populated the cache while this caller waited.
        # Reuse it instead of replacing files under that process's readers.
        if bundle_is_local():
            return root
        return _fetch_into_cache(root, verbose=verbose)


def status() -> dict:
    """Snapshot of cache state — used by ``oncoref data status bundle``."""
    root = cache_dir()
    items = _bundle_inventory(root)
    for p, item in items.items():
        item["path"] = str(root / p)
    marker = _read_completion_marker(root)
    contract = bundle_contract()
    return {
        "contract_version": contract["contract_version"],
        "package_version": __version__,
        "data_version": DATA_VERSION,
        "source_matrix_version": SOURCE_MATRIX_VERSION,
        "cache_dir": str(root),
        "release_url": RELEASE_URL,
        "release_urls": list(RELEASE_URLS),
        "release_manifest_url": RELEASE_MANIFEST_URL,
        "release_checksum_url": RELEASE_CHECKSUM_URL,
        "completion_marker": {
            "path": str(_completion_marker_path(root)),
            "present": marker is not None,
            "valid": _completion_marker_valid(root),
            "source_url": marker.get("source_url") if marker else None,
            "verified_sha256": bool(marker and marker.get("verified_sha256")),
        },
        "items": items,
        "all_local": bundle_is_local(),
        "contract": contract,
    }


def verify_local() -> dict:
    """Return :func:`status` only when the current cache has a valid completion marker.

    This is stricter than :func:`bundle_is_local`: legacy complete caches remain
    readable, but callers that need a post-fetch integrity boundary can require the
    marker written by current oncoref fetches.
    """
    snap = status()
    if not snap["all_local"]:
        missing = [p for p, item in snap["items"].items() if not item["complete"]]
        raise FileNotFoundError(
            f"oncoref data bundle is incomplete at {cache_dir()}: {', '.join(missing)}"
        )
    if not snap["completion_marker"]["valid"]:
        raise RuntimeError(
            "oncoref data bundle lacks a valid checksum-verified completion marker; "
            "run `oncoref data fetch bundle` to refresh it"
        )
    return snap


def _canonical_downloadable_path(relative_path: str) -> str | None:
    """Resolve extensionless CSV aliases to their physical bundle path."""
    parts = Path(relative_path).parts
    if not parts:
        return None
    first = parts[0].lower()
    for root in DOWNLOADABLE_PATHS:
        aliases = {
            root.lower(),
            root.removesuffix(".gz").lower(),
            root.removesuffix(".csv").lower(),
            root.removesuffix(".csv.gz").lower(),
        }
        if first in aliases:
            return str(Path(root, *parts[1:]))
    return None


def is_downloadable(relative_path: str) -> bool:
    """True if ``relative_path`` falls under one of the downloadable roots."""
    return _canonical_downloadable_path(relative_path) is not None


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def list_cache_versions() -> list[dict]:
    """Enumerate every version-pinned cache dir under :func:`cache_root`.

    Returns ``{"version", "path", "size_bytes", "is_current"}`` dicts sorted by
    version label. Used by ``oncoref data prune`` to find upgrade leftovers.
    """
    root = cache_root()
    if not root.exists():
        return []
    current = cache_dir()
    out: list[dict] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("v"):
            continue
        out.append(
            {
                "version": child.name,
                "path": child,
                "size_bytes": _dir_size_bytes(child),
                "is_current": child.resolve() == current.resolve(),
            }
        )
    return out


def prune_cache(*, keep_current: bool = True, dry_run: bool = False) -> list[dict]:
    """Delete every version-pinned cache dir EXCEPT the installed version (when
    ``keep_current=True``). With ``dry_run=True`` returns the candidate list
    without touching disk. Returns the list of dirs deleted (or planned)."""
    candidates = []
    for entry in list_cache_versions():
        if keep_current and entry["is_current"]:
            continue
        candidates.append(entry)
    if dry_run:
        return candidates
    for entry in candidates:
        path = entry["path"]
        for child in sorted(path.rglob("*"), reverse=True):
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink()
                else:
                    child.rmdir()
            except OSError:
                pass
        with contextlib.suppress(OSError):
            path.rmdir()
    return candidates


__all__ = [
    "BUNDLE_CONTRACT_VERSION",
    "BUNDLE_MANIFEST_VERSION",
    "CACHE_COMPLETE_FILENAME",
    "CHECKSUM_FILENAME",
    "DOWNLOADABLE_PATHS",
    "FALLBACK_CHECKSUM_FILENAME",
    "FALLBACK_GITHUB_REPO",
    "FALLBACK_MANIFEST_FILENAME",
    "FALLBACK_RELEASE_CHECKSUM_URL",
    "FALLBACK_RELEASE_MANIFEST_URL",
    "FALLBACK_RELEASE_URL",
    "FALLBACK_TARBALL_FILENAME",
    "GITHUB_REPO",
    "MANIFEST_FILENAME",
    "RELEASE_CHECKSUM_URL",
    "RELEASE_MANIFEST_URL",
    "RELEASE_SOURCES",
    "RELEASE_URL",
    "RELEASE_URLS",
    "TARBALL_FILENAME",
    "BundleIntegrityError",
    "bundle_contract",
    "bundle_is_local",
    "bundle_release_manifest",
    "cache_dir",
    "cache_root",
    "ensure_local",
    "fetch",
    "find",
    "find_local_item",
    "is_downloadable",
    "is_local",
    "item_is_local",
    "list_cache_versions",
    "local_item_paths",
    "prune_cache",
    "status",
    "verify_local",
]
