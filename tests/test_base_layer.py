# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""oncoref is the base of the dependency pyramid.

It is the upstream source of truth for cancer reference data (expression, HPA
protein/RNA, the CTA definition, the cancer-type ontology, anti-PD-1 ORR, TMB) and
must never import its consumers — data and logic flow only downward. This guard
fails the build if any consumer leaks into the shipped package.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "oncoref"
REPOSITORY_ROOT = PACKAGE_ROOT.parent

# Downstream consumers that depend on oncoref; importing any of them would
# invert the dependency pyramid.
DOWNSTREAM_PACKAGES = {"pirlygenes", "tsarina", "hitlist", "trufflepig"}
LEGACY_PACKAGE_NAMES = {"oncodata"}
FORBIDDEN_DEPENDENCIES = DOWNSTREAM_PACKAGES | LEGACY_PACKAGE_NAMES


def _imported_top_level_modules(path):
    # Static (AST) scan of `import`/`from ... import` statements. A dynamic import
    # (importlib.import_module("pirlygenes"), __import__(...)) would evade this —
    # acceptable for a guard, since a static import is the realistic regression.
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        # node.level == 0 skips relative (intra-package) imports.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def test_package_imports_no_downstream_or_legacy_package():
    offenders = []
    for py in sorted(PACKAGE_ROOT.rglob("*.py")):
        for mod in _imported_top_level_modules(py):
            if mod in FORBIDDEN_DEPENDENCIES:
                offenders.append(f"{py.relative_to(PACKAGE_ROOT)} imports {mod}")
    assert not offenders, (
        "oncoref must not import downstream or legacy packages:\n  " + "\n  ".join(offenders)
    )


def test_project_metadata_depends_on_no_downstream_or_legacy_package():
    """Check build, runtime, and optional requirements without a TOML dependency."""
    metadata = (REPOSITORY_ROOT / "pyproject.toml").read_text()
    dependency_sections = re.findall(
        r"(?ms)^\[(?:build-system|project|project\.optional-dependencies)\]\n(.*?)(?=^\[|\Z)",
        metadata,
    )
    declared_names = {
        re.split(r"[\s\[<>=!~;]", requirement, maxsplit=1)[0].lower()
        for section in dependency_sections
        for requirement in re.findall(r'^\s*"([^"]+)"', section)
    }
    forbidden_names = declared_names & FORBIDDEN_DEPENDENCIES

    assert not forbidden_names, (
        "oncoref metadata must not depend on downstream or legacy packages: "
        f"{sorted(forbidden_names)}"
    )


def test_migration_scripts_import_no_downstream_or_legacy_package():
    offenders = []
    for py in sorted((REPOSITORY_ROOT / "scripts").rglob("*.py")):
        for mod in _imported_top_level_modules(py):
            if mod in FORBIDDEN_DEPENDENCIES:
                offenders.append(f"{py.relative_to(REPOSITORY_ROOT)} imports {mod}")
    assert not offenders, "oncoref scripts must not import downstream or legacy packages:\n  " + (
        "\n  ".join(offenders)
    )


def test_import_oncoref_does_not_configure_root_logging():
    code = """
import logging
root = logging.getLogger()
assert len(root.handlers) == 0
assert root.level == logging.WARNING
import oncoref
assert len(root.handlers) == 0, root.handlers
assert root.level == logging.WARNING, logging.getLevelName(root.level)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout


def test_genome_release_probe_preserves_root_logging():
    code = """
import logging
from oncoref import genome
root = logging.getLogger()
assert len(root.handlers) == 0
assert root.level == logging.WARNING
try:
    genome.genomes()
except genome.GenomeDependencyError:
    pass
assert len(root.handlers) == 0, root.handlers
assert root.level == logging.WARNING, logging.getLevelName(root.level)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
