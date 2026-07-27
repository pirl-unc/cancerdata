#!/usr/bin/env bash
# Build the oncoref expression-data tarball for upload to a GitHub Release.
#
# The wheel ships only the small curated tables; the heavy per-cohort expression
# artifacts (data_bundle.DOWNLOADABLE_PATHS) are distributed as a version-pinned
# tarball attached to the pirl-unc/oncoref release. This script packages those
# paths from a source directory that already contains them — e.g. a populated
# cache dir (`oncoref data fetch bundle` then `oncoref data dir bundle` for the
# path) or a pirlygenes data checkout during the migration.
#
# Usage:
#   scripts/build_data_tarball.sh <source-dir> [output-dir]
#   ONCOREF_DATA_RELEASE_VERSION=5.23.13 \
#       scripts/build_data_tarball.sh <source-dir> [output-dir]
#
# Then: upload <output-dir>/oncoref-data-v<DATA_VERSION>.tar.gz plus the emitted
# .sha256 and .manifest.json files to the `v<DATA_VERSION>` release on
# pirl-unc/oncoref, and only THEN bump DATA_VERSION.
set -euo pipefail

SRC="${1:?usage: build_data_tarball.sh <source-dir> [output-dir]}"
OUT_DIR="${2:-.}"

# Resolve user-supplied paths to absolute BEFORE cd'ing into the repo, so a
# relative source/output dir stays relative to the caller's CWD.
SRC="$(cd "$SRC" 2>/dev/null && pwd)" || {
    echo "error: source dir '$1' not found" >&2
    exit 1
}
mkdir -p "${OUT_DIR%/}"
OUT_DIR="$(cd "${OUT_DIR%/}" && pwd)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CODE_DATA_VERSION="$(python -c 'from oncoref.version import DATA_VERSION; print(DATA_VERSION)')"
DATA_VERSION="${ONCOREF_DATA_RELEASE_VERSION:-$CODE_DATA_VERSION}"
if [[ ! "$DATA_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "error: data release version must have the form X.Y.Z, got '$DATA_VERSION'" >&2
    exit 1
fi
PACKAGE_VERSION="$(python -c 'from oncoref.version import __version__; print(__version__)')"
SOURCE_MATRIX_VERSION="$(python -c 'from oncoref.version import SOURCE_MATRIX_VERSION; print(SOURCE_MATRIX_VERSION)')"
BUILDER_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
read -r -a PATHS <<<"$(python -c 'from oncoref.data_bundle import DOWNLOADABLE_PATHS; print(" ".join(DOWNLOADABLE_PATHS))')"
OUT="${OUT_DIR%/}/oncoref-data-v${DATA_VERSION}.tar.gz"
SHA_OUT="${OUT}.sha256"
MANIFEST_OUT="${OUT_DIR%/}/oncoref-data-v${DATA_VERSION}.manifest.json"

missing=()
for p in "${PATHS[@]}"; do
    [[ -e "$SRC/$p" ]] || missing+=("$p")
done
if ((${#missing[@]})); then
    echo "error: source dir '$SRC' is missing required bundle paths:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

echo "packaging ${#PATHS[@]} bundle paths from $SRC -> $OUT"
COPYFILE_DISABLE=1 tar -czf "$OUT" -C "$SRC" "${PATHS[@]}"
SIZE="$(du -h "$OUT" | cut -f1)"
echo "wrote $OUT ($SIZE)"
python - "$OUT" "$SRC" "$MANIFEST_OUT" "$DATA_VERSION" "$PACKAGE_VERSION" "$SOURCE_MATRIX_VERSION" "$BUILDER_COMMIT" "${PATHS[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

tarball = Path(sys.argv[1])
source = Path(sys.argv[2])
manifest = Path(sys.argv[3])
data_version = sys.argv[4]
package_version = sys.argv[5]
source_matrix_version = sys.argv[6]
builder_commit = sys.argv[7] or None
paths = sys.argv[8:]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(rel: str) -> dict:
    path = source / rel
    files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
    return {
        "path": rel,
        "file_count": len(files),
        "size_bytes": sum(p.stat().st_size for p in files),
    }


payload = {
    "manifest_version": 1,
    "data_version": data_version,
    "package_version": package_version,
    "source_matrix_version": source_matrix_version,
    "builder": "scripts/build_data_tarball.sh",
    "builder_commit": builder_commit,
    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "tarball": {
        "filename": tarball.name,
        "bytes": tarball.stat().st_size,
        "sha256": sha256_file(tarball),
        "downloadable_paths": paths,
    },
    "inventory": {rel: inventory(rel) for rel in paths},
}

build_json = source / "expression-artifact-build-metadata.json"
if build_json.exists():
    build_metadata = json.loads(build_json.read_text())
    derived_artifacts = build_metadata.get("derived_artifacts") or []
    payload["sample_qc_policy"] = build_metadata.get("sample_qc")
    payload["sample_qc_policy_version"] = build_metadata.get("sample_qc_policy_version")
    payload["source_matrix_sample_qc"] = build_metadata.get("sample_qc_manifest")
    payload["artifact_build_metadata"] = {
        "cohort_metadata": build_metadata.get("cohort_metadata"),
        "bundle_metadata": build_json.name,
        "derived_artifacts": derived_artifacts,
        "released_derived_artifacts": [p for p in derived_artifacts if p in paths],
        "unreleased_intermediate_artifacts": [p for p in derived_artifacts if p not in paths],
        "n_cohorts": build_metadata.get("n_cohorts"),
        "n_source_samples": build_metadata.get("n_source_samples"),
        "n_cohort_samples": build_metadata.get("n_cohort_samples"),
        "sample_qc_fallbacks": build_metadata.get("sample_qc_fallbacks"),
        "n_negative_values_clipped": build_metadata.get("n_negative_values_clipped"),
        "representative_partition": build_metadata.get("representative_partition"),
    }
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
python - "$OUT" "$SHA_OUT" <<'PY'
import hashlib
import sys
from pathlib import Path

tarball = Path(sys.argv[1])
out = Path(sys.argv[2])
h = hashlib.sha256()
with tarball.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(chunk)
out.write_text(f"{h.hexdigest()}  {tarball.name}\n")
PY
echo "wrote $SHA_OUT"
echo "wrote $MANIFEST_OUT"
echo
echo "next: gh release upload v${DATA_VERSION} '$OUT' '$SHA_OUT' '$MANIFEST_OUT' --repo pirl-unc/oncoref"
echo "      (create the v${DATA_VERSION} release first if it doesn't exist)"
