#!/usr/bin/env python
"""Classify whether a GitHub release should publish the oncoref package."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

DEPLOY_PYPI_PUBLISHED_MARKER = "<!-- oncoref-deploy-pypi-published -->"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_VERSION_PATH = _REPO_ROOT / "oncoref" / "version.py"


def should_publish_to_pypi(
    *,
    release_tag: str,
    release_body: str,
    package_version: str,
) -> bool:
    """Return whether this is an unpublished release of the package version."""
    package_tag = f"v{package_version}"
    already_published = DEPLOY_PYPI_PUBLISHED_MARKER in release_body
    return release_tag == package_tag and not already_published


def package_version_from_source() -> str:
    """Read the package version without importing oncoref or its dependencies."""
    return str(runpy.run_path(str(_VERSION_PATH))["__version__"])


def write_github_output() -> None:
    """Write the release decision in GitHub Actions job-output format."""
    release_tag = os.environ.get("RELEASE_TAG", "")
    release_body = os.environ.get("RELEASE_BODY", "")
    publish = should_publish_to_pypi(
        release_tag=release_tag,
        release_body=release_body,
        package_version=package_version_from_source(),
    )

    output_path = Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open("a", encoding="utf-8") as output:
        output.write(f"publish={str(publish).lower()}\n")


if __name__ == "__main__":
    write_github_output()
