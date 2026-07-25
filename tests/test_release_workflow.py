from pathlib import Path


def test_build_backend_supports_declared_pep_639_license_metadata():
    pyproject = Path("pyproject.toml").read_text()

    assert 'requires = ["setuptools>=77.0.0", "wheel"]' in pyproject
    assert 'license = "Apache-2.0"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject


def test_release_workflow_skips_deploy_sh_created_releases():
    workflow = Path(".github/workflows/release.yml").read_text()

    assert "oncoref-deploy-pypi-published" in workflow
    assert "gh-action-pypi-publish" in workflow


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
