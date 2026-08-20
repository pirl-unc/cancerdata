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

"""Versioned reference-data sources fetched on demand (Human Protein Atlas).

These are per-file downloads (each a ``.zip`` that extracts to one ``.tsv``),
distinct from the heavy per-cohort expression bundle (:mod:`oncoref.data_bundle`,
a single tarball). They back the CTA tissue-restriction definition and the
protein-level / single-cell normal-tissue comparisons.

Pinned to HPA ``v23`` — the most recent release whose mirror serves RNA consensus
AND IHC ``normal_tissue`` as a matched pair (newer mirrors drop ``normal_tissue``).

Cache layout (one subdir per source+version):

    <cache>/sources/<name>/<version>/<filename>

The cache root is ``~/.cache/oncoref/sources`` (override with
``CANCERDATA_DATA_DIR``). A ``manifest.json`` records URL / size / sha256 /
download time for provenance.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

#: Default HPA release (matched RNA + IHC pair).
DEFAULT_HPA_VERSION = "v23"

#: Registry of reference sources. ``urls`` maps a version label to a download
#: URL; ``.zip`` URLs are transparently extracted to ``filename``.
REFERENCE_SOURCES: dict[str, dict] = {
    "hpa_rna_consensus": {
        "description": "HPA RNA consensus tissue nTPM (per-tissue normal expression)",
        "filename": "rna_tissue_consensus.tsv",
        "urls": {
            "v23": "https://v23.proteinatlas.org/download/rna_tissue_consensus.tsv.zip",
            "latest": "https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip",
        },
    },
    "hpa_normal_tissue": {
        "description": "HPA IHC protein expression (normal_tissue, per tissue/cell type)",
        "filename": "normal_tissue.tsv",
        "urls": {
            "v23": "https://v23.proteinatlas.org/download/normal_tissue.tsv.zip",
        },
    },
    "hpa_single_cell": {
        "description": "HPA single-cell type RNA nTPM (per cell-type normal expression)",
        "filename": "rna_single_cell_type.tsv",
        "urls": {
            "v23": "https://v23.proteinatlas.org/download/rna_single_cell_type.tsv.zip",
            "latest": "https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip",
        },
    },
}

#: Env var overriding the reference-data cache root.
CACHE_DIR_ENV_KEY = "CANCERDATA_DATA_DIR"


class ReferenceDataError(RuntimeError):
    """Raised for unknown sources/versions or download failures."""


def cache_dir() -> Path:
    """Reference-data cache directory (``<root>/sources``), created on demand."""
    override = os.environ.get(CACHE_DIR_ENV_KEY)
    base = Path(override).expanduser() if override else Path.home() / ".cache" / "oncoref"
    sources = base / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    return sources


def _source(name: str) -> dict:
    try:
        return REFERENCE_SOURCES[name]
    except KeyError:
        avail = ", ".join(sorted(REFERENCE_SOURCES))
        raise ReferenceDataError(f"unknown reference source {name!r}; available: {avail}") from None


def resolve_version(name: str, version: str | None = None) -> str:
    """Concrete version for *name* (defaults to the pinned HPA release)."""
    spec = _source(name)
    if version is None:
        version = DEFAULT_HPA_VERSION
    if version not in spec["urls"]:
        avail = ", ".join(sorted(spec["urls"]))
        raise ReferenceDataError(f"{name!r} has no version {version!r}; available: {avail}")
    return version


def local_path(name: str, version: str | None = None) -> Path:
    """Expected cache path for *name*/*version* (may not exist yet)."""
    version = resolve_version(name, version)
    spec = _source(name)
    return cache_dir() / name / version / spec["filename"]


def is_cached(name: str, version: str | None = None) -> bool:
    return local_path(name, version).exists()


def _manifest_path() -> Path:
    return cache_dir() / "manifest.json"


def _manifest_key(name: str, version: str) -> str:
    """Manifest record key. Keyed by ``(name, version)`` so each cached version of
    a multi-version source (e.g. ``hpa_rna_consensus`` v23 + latest) keeps its own
    size/sha256 record — a single per-name record would only validate whichever
    version was downloaded last."""
    return f"{name}@{version}"


def _manifest_record(name: str, version: str) -> dict:
    """The manifest record for ``(name, version)``. Falls back to a legacy
    per-name record (pre-versioned-key manifests) so an existing cache keeps its
    provenance until the next download rewrites it in the new layout."""
    manifest = _read_manifest()
    record = manifest.get(_manifest_key(name, version))
    if record is not None:
        return record
    legacy = manifest.get(name) or {}
    return legacy if legacy.get("version") == version else {}


def _read_manifest() -> dict:
    path = _manifest_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_manifest(manifest: dict) -> None:
    # provenance manifest is best-effort
    with contextlib.suppress(OSError):
        _manifest_path().write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _cached_file_ok(name: str, version: str, dest: Path) -> bool:
    """Cheap reuse check: trust an existing cached file only if its size matches
    the size recorded in the manifest for this ``(name, version)`` (when one
    exists). Catches a TSV left truncated by a process killed mid-extract — a
    partial write that the old ``exists()``-only reuse would have served forever
    (issue #21). A file with no manifest record is trusted (can't disprove;
    preserves back-compat). The full content hash is checked only on explicit
    :func:`verify` to avoid re-hashing a large TSV on every access."""
    expected = _manifest_record(name, version).get("bytes")
    if expected is None:
        return True
    try:
        return dest.stat().st_size == int(expected)
    except OSError:
        return False


def download(name: str, version: str | None = None, *, force: bool = False) -> Path:
    """Download *name*/*version* into the cache (extracting the ``.zip``) and
    record it in the manifest. A cached copy is reused unless ``force=True`` —
    or it fails the manifest size check (a truncated/partial cache), in which
    case it is re-fetched."""
    version = resolve_version(name, version)
    spec = _source(name)
    dest = local_path(name, version)
    url = spec["urls"][version]

    if dest.exists() and not force and _cached_file_ok(name, version, dest):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = dest.parent / (spec["filename"] + ".zip.part")
    tmp_tsv = dest.parent / (spec["filename"] + ".part")
    try:
        sys.stderr.write(f"oncoref: downloading {name} ({url})\n")
        sys.stderr.flush()
        with urllib.request.urlopen(url) as resp, tmp_zip.open("wb") as h:
            shutil.copyfileobj(resp, h, length=1024 * 1024)
        with zipfile.ZipFile(tmp_zip) as zf:
            member = _zip_member(zf, spec["filename"])
            with zf.open(member) as src, tmp_tsv.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
        # Atomic promote: a process killed mid-extract leaves tmp_tsv, never a
        # partial dest. A prior good copy survives a failed re-download untouched.
        os.replace(tmp_tsv, dest)
    except Exception as e:  # network / zip / IO — surface uniformly
        raise ReferenceDataError(f"failed to download {name} ({url}): {e}") from e
    finally:
        tmp_zip.unlink(missing_ok=True)
        tmp_tsv.unlink(missing_ok=True)

    manifest = _read_manifest()
    manifest[_manifest_key(name, version)] = {
        "name": name,
        "version": version,
        "url": url,
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_manifest(manifest)
    return dest


def verify(name: str, version: str | None = None) -> bool:
    """Full-content integrity check of a cached source against the manifest sha256.

    Returns ``True`` when the cached file's sha256 matches what the manifest
    recorded at download time, ``False`` on mismatch or when the file is missing.
    Raises if there is no manifest record to check against (nothing to verify).
    Unlike the cheap size check on reuse, this re-hashes the file — call it
    deliberately (a cache audit), not on every access."""
    record = provenance(name, version, verify_content=True)
    if not record["sha256"]:
        raise ReferenceDataError(
            f"no manifest sha256 recorded for {name!r} {record['version']}; nothing to verify"
        )
    return record["checksum_matches"] is True


def provenance(
    name: str,
    version: str | None = None,
    *,
    verify_content: bool = False,
) -> dict:
    """Return source and cache provenance for one concrete reference artifact.

    The returned dictionary is a defensive, scalar-only snapshot; changing it
    cannot mutate the cache manifest. ``verify_content=False`` avoids hashing
    large HPA TSVs and reports ``verification_state="not_checked"`` when a
    checksum is available. Opting into content verification reports one of
    ``"verified"`` or ``"checksum_mismatch"``. The other explicit states are
    ``"no_manifest"``, ``"missing_checksum"``, and ``"missing_file"``.
    """
    version = resolve_version(name, version)
    spec = _source(name)
    dest = local_path(name, version)
    record = dict(_manifest_record(name, version))
    exists = dest.exists()
    try:
        local_bytes = dest.stat().st_size if exists else None
    except OSError:
        exists = False
        local_bytes = None

    recorded_bytes = record.get("bytes")
    try:
        recorded_bytes = int(recorded_bytes) if recorded_bytes is not None else None
    except (TypeError, ValueError):
        recorded_bytes = None

    expected = record.get("sha256") or None
    checksum_matches = None
    if not record:
        verification_state = "no_manifest"
    elif not expected:
        verification_state = "missing_checksum"
    elif not exists:
        verification_state = "missing_file"
        checksum_matches = False
    elif verify_content:
        checksum_matches = hashlib.sha256(dest.read_bytes()).hexdigest() == expected
        verification_state = "verified" if checksum_matches else "checksum_mismatch"
    else:
        verification_state = "not_checked"

    return {
        "name": name,
        "version": version,
        "url": record.get("url") or spec["urls"][version],
        "path": str(dest),
        "bytes": local_bytes,
        "recorded_bytes": recorded_bytes,
        "sha256": expected,
        "downloaded_at": record.get("downloaded_at") or None,
        "exists": exists,
        "manifest_recorded": bool(record),
        "checksum_matches": checksum_matches,
        "verification_state": verification_state,
    }


def _zip_member(zf: zipfile.ZipFile, preferred: str) -> str:
    """Pick the .tsv member to extract (preferred name, else the first .tsv)."""
    names = zf.namelist()
    if preferred in names:
        return preferred
    tsvs = [n for n in names if n.endswith(".tsv")]
    if not tsvs:
        raise ReferenceDataError(f"no .tsv found in archive (members: {names})")
    return tsvs[0]


def ensure(name: str, version: str | None = None) -> Path:
    """Return a local path to *name*/*version*, downloading if absent or if the
    cached copy fails the manifest size check (a truncated/partial file)."""
    version = resolve_version(name, version)
    path = local_path(name, version)
    if path.exists() and _cached_file_ok(name, version, path):
        return path
    return download(name, version)


def status(*, verify_content: bool = False) -> list[dict]:
    """One row per source with default-version cache and provenance status.

    Set ``verify_content=True`` to hash each present artifact and populate a
    definitive checksum verification state. The default is deliberately cheap.
    """
    rows = []
    for name, spec in REFERENCE_SOURCES.items():
        source_provenance = provenance(name, verify_content=verify_content)
        rows.append(
            {
                **source_provenance,
                "cached": source_provenance["exists"],
                "bytes": source_provenance["bytes"] or 0,
                "default_version": DEFAULT_HPA_VERSION,
                "available_versions": sorted(spec["urls"]),
                "cached_version": (
                    source_provenance["version"] if source_provenance["manifest_recorded"] else None
                ),
                "description": spec["description"],
            }
        )
    return rows
