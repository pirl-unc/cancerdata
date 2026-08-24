#!/usr/bin/env python3
"""Build a deterministic, checksum-pinned data-bundle overlay release.

The output archive contains only files that differ from a published complete
base bundle. Its manifest retains the complete final inventory and pins both
the base archive and overlay archive by byte size and SHA-256.

Usage:
    python scripts/build_data_overlay.py COMPLETE_DIR BASE_DIR BASE_MANIFEST OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.data_bundle import DOWNLOADABLE_PATHS
from oncoref.version import DATA_VERSION, SOURCE_MATRIX_VERSION, __version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for relative in DOWNLOADABLE_PATHS:
        path = root / relative
        if not path.exists():
            raise SystemExit(f"error: {root} is missing required bundle path {relative!r}")
        members = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        if not members:
            raise SystemExit(f"error: required bundle path {relative!r} is empty in {root}")
        for member in members:
            files[member.relative_to(root).as_posix()] = member
    return files


def same_file(left: Path, right: Path) -> bool:
    left_stat = left.stat()
    right_stat = right.stat()
    if left_stat.st_dev == right_stat.st_dev and left_stat.st_ino == right_stat.st_ino:
        return True
    if left_stat.st_size != right_stat.st_size:
        return False
    return sha256_file(left) == sha256_file(right)


def inventory(root: Path, relative: str) -> dict:
    path = root / relative
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    return {
        "path": relative,
        "file_count": len(files),
        "size_bytes": sum(p.stat().st_size for p in files),
    }


def validate_base_manifest(payload: dict) -> tuple[str, str, dict]:
    base_version = str(payload.get("data_version") or "")
    if len(base_version.split(".")) != 3 or not all(
        part.isdigit() for part in base_version.split(".")
    ):
        raise SystemExit(f"error: base manifest has invalid data_version {base_version!r}")
    if base_version == DATA_VERSION:
        raise SystemExit("error: overlay base version must differ from the release version")
    base_source_matrix_version = str(payload.get("source_matrix_version") or "")
    if len(base_source_matrix_version.split(".")) != 3 or not all(
        part.isdigit() for part in base_source_matrix_version.split(".")
    ):
        raise SystemExit(
            f"error: base manifest has invalid source_matrix_version {base_source_matrix_version!r}"
        )
    tarball = payload.get("tarball")
    if not isinstance(tarball, dict):
        raise SystemExit("error: base manifest lacks tarball metadata")
    expected_filename = f"oncoref-data-v{base_version}.tar.gz"
    if tarball.get("filename") != expected_filename:
        raise SystemExit(
            f"error: base manifest tarball must be {expected_filename!r}, "
            f"got {tarball.get('filename')!r}"
        )
    checksum = str(tarball.get("sha256") or "").lower()
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise SystemExit("error: base manifest lacks a valid tarball SHA-256")
    try:
        size_bytes = int(tarball.get("bytes"))
    except (TypeError, ValueError) as error:
        raise SystemExit("error: base manifest lacks a valid tarball byte size") from error
    if tuple(tarball.get("downloadable_paths") or ()) != DOWNLOADABLE_PATHS:
        raise SystemExit("error: base manifest downloadable paths do not match oncoref")
    return (
        base_version,
        base_source_matrix_version,
        {
            "filename": expected_filename,
            "bytes": size_bytes,
            "sha256": checksum,
            "downloadable_paths": list(DOWNLOADABLE_PATHS),
        },
    )


def write_deterministic_archive(output: Path, files: dict[str, Path], paths: list[str]) -> None:
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for relative in paths:
            source = files[relative]
            info = archive.gettarinfo(str(source), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with source.open("rb") as handle:
                archive.addfile(info, handle)


def add_build_metadata(payload: dict, complete: Path) -> None:
    build_json = complete / "expression-artifact-build-metadata.json"
    if not build_json.exists():
        return
    build_metadata = json.loads(build_json.read_text())
    derived_artifacts = build_metadata.get("derived_artifacts") or []
    payload["sample_qc_policy"] = build_metadata.get("sample_qc")
    payload["sample_qc_policy_version"] = build_metadata.get("sample_qc_policy_version")
    payload["source_matrix_sample_qc"] = build_metadata.get("sample_qc_manifest")
    payload["artifact_build_metadata"] = {
        "cohort_metadata": build_metadata.get("cohort_metadata"),
        "bundle_metadata": build_json.name,
        "derived_artifacts": derived_artifacts,
        "released_derived_artifacts": [
            path for path in derived_artifacts if path in DOWNLOADABLE_PATHS
        ],
        "unreleased_intermediate_artifacts": [
            path for path in derived_artifacts if path not in DOWNLOADABLE_PATHS
        ],
        "n_cohorts": build_metadata.get("n_cohorts"),
        "n_source_samples": build_metadata.get("n_source_samples"),
        "n_cohort_samples": build_metadata.get("n_cohort_samples"),
        "sample_qc_fallbacks": build_metadata.get("sample_qc_fallbacks"),
        "n_negative_values_clipped": build_metadata.get("n_negative_values_clipped"),
        "representative_partition": build_metadata.get("representative_partition"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("complete_dir", type=Path)
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    complete = args.complete_dir.resolve(strict=True)
    base = args.base_dir.resolve(strict=True)
    base_payload = json.loads(args.base_manifest.read_text())
    base_version, base_source_matrix_version, base_tarball = validate_base_manifest(base_payload)
    data_version = os.environ.get("ONCOREF_DATA_RELEASE_VERSION", DATA_VERSION)
    if len(data_version.split(".")) != 3 or not all(
        part.isdigit() for part in data_version.split(".")
    ):
        raise SystemExit(f"error: invalid release version {data_version!r}")

    complete_files = bundle_files(complete)
    base_files = bundle_files(base)
    changed = sorted(
        relative
        for relative, path in complete_files.items()
        if relative not in base_files or not same_file(path, base_files[relative])
    )
    deleted = sorted(set(base_files) - set(complete_files))
    if not changed and not deleted:
        raise SystemExit("error: complete bundle does not differ from its base")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / f"oncoref-data-v{data_version}.tar.gz"
    manifest = args.output_dir / f"oncoref-data-v{data_version}.manifest.json"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    write_deterministic_archive(archive, complete_files, changed)

    builder_commit = None
    with contextlib.suppress(OSError, subprocess.CalledProcessError):
        builder_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    payload = {
        "manifest_version": 2,
        "bundle_layout": "overlay",
        "data_version": data_version,
        "package_version": __version__,
        "source_matrix_version": SOURCE_MATRIX_VERSION,
        "builder": "scripts/build_data_overlay.py",
        "builder_commit": builder_commit,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_bundle": {
            "data_version": base_version,
            "source_matrix_version": base_source_matrix_version,
            "repo": "pirl-unc/oncoref",
            "tarball": base_tarball,
        },
        "overlay": {
            "file_count": len(changed),
            "size_bytes": sum(complete_files[path].stat().st_size for path in changed),
            "paths": changed,
            "deleted_paths": deleted,
        },
        "tarball": {
            "filename": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "downloadable_paths": list(DOWNLOADABLE_PATHS),
        },
        "inventory": {path: inventory(complete, path) for path in DOWNLOADABLE_PATHS},
    }
    add_build_metadata(payload, complete)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    checksum.write_text(f"{payload['tarball']['sha256']}  {archive.name}\n")

    print(
        f"wrote {archive} ({archive.stat().st_size:,} bytes; "
        f"{len(changed)} changed files, {len(deleted)} deletions)"
    )
    print(f"wrote {checksum}")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
