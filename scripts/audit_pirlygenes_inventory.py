#!/usr/bin/env python3
"""Reproduce or compare the pinned pirlygenes migration inventory.

This is a developer audit; oncoref has no runtime dependency on pirlygenes.
Point ``--repo`` at a local clone and optionally select another ``--ref`` to
review downstream inventory drift under the current ownership boundary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref import data_manifest


def normalize_inventory_entry(name: str) -> str:
    """Normalize one direct ``pirlygenes/data`` entry to its dataset name."""

    for suffix in (".csv.gz", ".csv", ".yaml", ".yml"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def inventory_from_entries(entries) -> frozenset[str]:
    return frozenset(normalize_inventory_entry(str(entry).strip()) for entry in entries if entry)


def inventory_from_git(repo: Path, ref: str, data_path: str) -> frozenset[str]:
    command = ["git", "-C", str(repo), "ls-tree", "--name-only", f"{ref}:{data_path}"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return inventory_from_entries(result.stdout.splitlines())


def inventory_diff(expected: frozenset[str], observed: frozenset[str]) -> dict[str, list[str]]:
    return {
        "added": sorted(observed - expected),
        "removed": sorted(expected - observed),
    }


def main(argv=None) -> int:
    metadata = data_manifest.PIRLYGENES_MIGRATION_SNAPSHOT_METADATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "pirlygenes",
        help="local pirlygenes clone (default: sibling of oncoref)",
    )
    parser.add_argument(
        "--ref",
        default=metadata["commit"],
        help="git ref to audit (default: pinned migration commit)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(argv)

    observed = inventory_from_git(args.repo, args.ref, metadata["data_path"])
    expected = data_manifest.PIRLYGENES_MIGRATION_SNAPSHOT
    diff = inventory_diff(expected, observed)
    payload = {
        "repository": metadata["repository"],
        "repo_path": str(args.repo.resolve()),
        "ref": args.ref,
        "pinned_ref": metadata["ref"],
        "pinned_commit": metadata["commit"],
        "expected_count": len(expected),
        "observed_count": len(observed),
        **diff,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"pirlygenes inventory at {args.ref}: {len(observed)} datasets")
        print(f"pinned migration snapshot: {len(expected)} datasets")
        print(f"added ({len(diff['added'])}): {', '.join(diff['added']) or '-'}")
        print(f"removed ({len(diff['removed'])}): {', '.join(diff['removed']) or '-'}")
    return 1 if diff["added"] or diff["removed"] else 0


if __name__ == "__main__":
    sys.exit(main())
