import re
from pathlib import Path

import yaml

NODE24_ACTION_MAJORS = {
    "actions/cache": 5,
    "actions/checkout": 6,
    "actions/configure-pages": 6,
    "actions/deploy-pages": 5,
    "actions/setup-python": 6,
    "actions/upload-pages-artifact": 5,
}


def test_workflows_use_node24_action_generations():
    observed = {action: [] for action in NODE24_ACTION_MAJORS}

    for path in sorted(Path(".github/workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        for job_name, job in workflow["jobs"].items():
            for step in job.get("steps", []):
                uses = step.get("uses", "")
                action, separator, version = uses.partition("@")
                if action not in observed:
                    continue
                assert separator, f"{path}:{job_name} has an unversioned {action} action"
                observed[action].append((path, job_name, version))

    for action, minimum_major in NODE24_ACTION_MAJORS.items():
        assert observed[action], f"no workflow uses expected action {action}"
        for path, job_name, version in observed[action]:
            match = re.fullmatch(r"v(\d+)(?:\.\d+\.\d+)?", version)
            assert match, f"{path}:{job_name} uses an unrecognized {action} version: {version}"
            actual_major = int(match.group(1))
            assert actual_major >= minimum_major, (
                f"{path}:{job_name} uses {action}@{version}; "
                f"Node 24 requires v{minimum_major} or newer"
            )


def test_test_workflow_stages_required_luad_matrix():
    workflow = yaml.safe_load(Path(".github/workflows/tests.yml").read_text())
    test_job = workflow["jobs"]["test"]

    assert "CANCERDATA_SOURCE_MATRICES" in test_job["env"]
    expected_sha256 = "001ebb9ad18aea86599ff5d808851007523f6d8f0927dfdf2a2f39372f83ffc5"
    assert test_job["env"]["CI_LUAD_SHA256"] == expected_sha256
    assert test_job["env"]["CI_LUAD_SOURCE_URL"].endswith(
        "/source-v5.22.8/LUAD_per_sample_tpm.parquet"
    )
    cache_step = next(
        step for step in test_job["steps"] if step["name"] == "Cache required source matrices"
    )
    assert cache_step["with"]["key"] == "source-matrices-luad-${{ env.CI_LUAD_SHA256 }}"

    stage_step = next(
        step for step in test_job["steps"] if step["name"] == "Stage required source matrices"
    )
    assert 'source_matrices.local_path("LUAD")' in stage_step["run"]
    assert 'os.environ["CI_LUAD_SOURCE_URL"]' in stage_step["run"]
    assert 'os.environ["CI_LUAD_SHA256"]' in stage_step["run"]
