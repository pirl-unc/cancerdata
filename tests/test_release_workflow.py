import json
import os
import subprocess
from pathlib import Path

import yaml
from scripts.pypi_release_gate import (
    DEPLOY_PYPI_PUBLISHED_MARKER,
    package_version_from_source,
    should_publish_to_pypi,
    write_github_output,
)

from oncoref.data_bundle import DOWNLOADABLE_PATHS


def test_build_backend_supports_declared_pep_639_license_metadata():
    pyproject = Path("pyproject.toml").read_text()

    assert 'requires = ["setuptools>=77.0.0", "wheel"]' in pyproject
    assert 'license = "Apache-2.0"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject


def test_source_distribution_includes_regeneration_scripts():
    manifest = Path("MANIFEST.in").read_text()

    assert "recursive-include scripts *.py" in manifest


def test_release_classifier_accepts_only_unpublished_package_tag():
    package_version = "1.2.3"
    cases = [
        ("v1.2.3", "Package release", True),
        ("v1.2.3", DEPLOY_PYPI_PUBLISHED_MARKER, False),
        ("v5.23.12", "Data bundle 5.23.12", False),
        ("source-v5.22.8", "Source matrices 5.22.8", False),
        ("v1.2.2", "Older package release", False),
    ]

    for release_tag, release_body, expected in cases:
        assert (
            should_publish_to_pypi(
                release_tag=release_tag,
                release_body=release_body,
                package_version=package_version,
            )
            is expected
        )


def test_release_workflow_gates_pypi_job_before_build():
    workflow = yaml.safe_load(Path(".github/workflows/release.yml").read_text())
    jobs = workflow["jobs"]
    classifier = jobs["classify-release"]
    publisher = jobs["build-and-publish"]
    classify_step = next(step for step in classifier["steps"] if step.get("id") == "classify")

    assert classifier["outputs"]["publish"] == "${{ steps.classify.outputs.publish }}"
    assert classify_step["run"] == "python scripts/pypi_release_gate.py"
    assert publisher["needs"] == "classify-release"
    assert publisher["if"] == "${{ needs.classify-release.outputs.publish == 'true' }}"
    assert publisher["permissions"] == {"contents": "read", "id-token": "write"}
    assert any(
        step.get("uses") == "pypa/gh-action-pypi-publish@release/v1" for step in publisher["steps"]
    )


def test_release_classifier_writes_github_job_output(monkeypatch, tmp_path):
    output_path = tmp_path / "github-output"
    package_tag = f"v{package_version_from_source()}"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setenv("RELEASE_TAG", package_tag)
    monkeypatch.setenv("RELEASE_BODY", "Package release")

    write_github_output()
    assert output_path.read_text() == "publish=true\n"

    monkeypatch.setenv("RELEASE_BODY", DEPLOY_PYPI_PUBLISHED_MARKER)
    write_github_output()
    assert output_path.read_text() == "publish=true\npublish=false\n"


def test_deploy_uses_one_virtualenv_python_interpreter():
    script = Path("deploy.sh").read_text()

    assert 'PYTHON_COMMAND="${PYTHON:-python}"' in script
    assert "sys.prefix != sys.base_prefix" in script
    assert 'export PATH="$(dirname "$PYTHON_BIN"):$PATH"' in script
    assert 'VERSION="$("$PYTHON_BIN" -c' in script
    assert '"$PYTHON_BIN" -m pip install --upgrade build twine' in script
    assert '"$PYTHON_BIN" -m build' in script
    assert '"$PYTHON_BIN" -m twine upload' in script

    command_lines = [line.strip() for line in script.splitlines()]
    assert not any(line.startswith(("python ", "python3 ")) for line in command_lines)


def test_data_tarball_can_target_an_unreleased_version(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for relative_path in DOWNLOADABLE_PATHS:
        path = source / relative_path
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}" if path.suffix == ".json" else "fixture\n")
        else:
            path.mkdir(parents=True)
            (path / "fixture.txt").write_text("fixture\n")

    env = os.environ.copy()
    env["ONCOREF_DATA_RELEASE_VERSION"] = "9.8.7"
    subprocess.run(
        ["bash", "scripts/build_data_tarball.sh", str(source), str(output)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    first_tarball = (output / "oncoref-data-v9.8.7.tar.gz").read_bytes()
    subprocess.run(
        ["bash", "scripts/build_data_tarball.sh", str(source), str(output)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    manifest = json.loads((output / "oncoref-data-v9.8.7.manifest.json").read_text())
    assert manifest["data_version"] == "9.8.7"
    assert manifest["tarball"]["filename"] == "oncoref-data-v9.8.7.tar.gz"
    assert (output / "oncoref-data-v9.8.7.tar.gz").read_bytes() == first_tarball
    assert (output / "oncoref-data-v9.8.7.tar.gz.sha256").exists()
