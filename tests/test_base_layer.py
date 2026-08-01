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

_PACKAGE = Path(__file__).resolve().parents[1] / "oncoref"
_PROJECT = _PACKAGE.parent

# Downstream consumers that depend on oncoref; importing any of them would
# invert the dependency pyramid.
_CONSUMERS = {"pirlygenes", "tsarina", "hitlist", "trufflepig"}


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


def test_package_imports_no_consumer():
    offenders = []
    for py in sorted(_PACKAGE.rglob("*.py")):
        for mod in _imported_top_level_modules(py):
            if mod in _CONSUMERS:
                offenders.append(f"{py.relative_to(_PACKAGE)} imports {mod}")
    assert not offenders, "oncoref must not import its consumers:\n  " + "\n  ".join(offenders)


def test_project_metadata_depends_on_no_consumer():
    """Keep downstream projects out of build, runtime, and optional requirements."""
    metadata = (_PROJECT / "pyproject.toml").read_text()
    dependency_sections = re.findall(
        r"(?ms)^\[(?:build-system|project|project\.optional-dependencies)\]\n(.*?)(?=^\[|\Z)",
        metadata,
    )
    declared_names = {
        re.split(r"[\s\[<>=!~;]", requirement, maxsplit=1)[0].lower()
        for section in dependency_sections
        for requirement in re.findall(r'^\s*"([^"]+)"', section)
    }

    assert declared_names.isdisjoint(_CONSUMERS), (
        "oncoref metadata must not depend on downstream consumers: "
        f"{sorted(declared_names & _CONSUMERS)}"
    )


def test_migration_scripts_import_no_consumer():
    offenders = []
    for py in sorted((_PROJECT / "scripts").rglob("*.py")):
        for mod in _imported_top_level_modules(py):
            if mod in _CONSUMERS:
                offenders.append(f"{py.relative_to(_PROJECT)} imports {mod}")
    assert not offenders, "oncoref migration scripts must not import consumers:\n  " + "\n  ".join(
        offenders
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
        cwd=_PROJECT,
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
        cwd=_PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
