import re
import shlex
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


def test_test_workflow_stages_required_real_data():
    workflow = yaml.safe_load(Path(".github/workflows/tests.yml").read_text())
    test_job = workflow["jobs"]["test"]

    assert "CANCERDATA_SOURCE_MATRICES" in test_job["env"]
    expected_sha256 = "001ebb9ad18aea86599ff5d808851007523f6d8f0927dfdf2a2f39372f83ffc5"
    assert test_job["env"]["CI_LUAD_SHA256"] == expected_sha256
    assert test_job["env"]["CI_LUAD_SOURCE_URL"].endswith(
        "/source-v5.22.8/LUAD_per_sample_tpm.parquet"
    )
    assert test_job["env"]["CI_ACC_SHA256"] == (
        "c871d0255d4aea60b55069da4eddbcea3d45a7e7b21358e77b829368df5b54f5"
    )
    assert test_job["env"]["CI_ACC_SOURCE_URL"].endswith(
        "/source-v5.22.8/ACC_per_sample_tpm.parquet"
    )
    assert test_job["env"]["CI_ACC_REFERENCE_SHA256"] == (
        "eab4da92aa12067f9f51d14532d3629d4a91732f9a3eb2fb03eb26f9f5fa4658"
    )
    assert test_job["env"]["CI_ACC_REFERENCE_URL"].endswith(
        "/ci-fixtures-v1/ACC_reference_percentiles_v5.23.2.parquet"
    )
    assert "CANCERDATA_DATA_DIR" in test_job["env"]
    assert test_job["env"]["CI_HPA_RNA_SHA256"] == (
        "c49b5b33a076e3b9c1eb4f7d15d063ee936e07045e192ceccfeb9edcf1d90ddb"
    )
    assert test_job["env"]["CI_HPA_NORMAL_TISSUE_SHA256"] == (
        "1fa9111070f23290d29a32eaa30695689599b5231e7d0bc935b60a777ad3a1cc"
    )
    cache_step = next(
        step for step in test_job["steps"] if step["name"] == "Cache required source matrices"
    )
    assert cache_step["with"]["key"] == (
        "source-matrices-${{ env.CI_LUAD_SHA256 }}-${{ env.CI_ACC_SHA256 }}"
    )

    stage_step = next(
        step for step in test_job["steps"] if step["name"] == "Stage required source matrices"
    )
    assert 'for code in ("LUAD", "ACC")' in stage_step["run"]
    assert "source_matrices.local_path(code)" in stage_step["run"]
    assert 'os.environ[f"CI_{code}_SOURCE_URL"]' in stage_step["run"]
    assert 'os.environ[f"CI_{code}_SHA256"]' in stage_step["run"]

    reference_step = next(
        step for step in test_job["steps"] if step["name"] == "Stage required reference data"
    )
    assert 'os.environ["ONCOREF_CI_ACC_PERCENTILE_REFERENCE"]' in reference_step["run"]
    assert "import shutil" in reference_step["run"]
    assert '"cancer-reference-expression-percentiles"' in reference_step["run"]
    assert "shutil.copy2(acc_reference, bundle_acc_reference)" in reference_step["run"]
    assert '"hpa_rna_consensus": os.environ["CI_HPA_RNA_SHA256"]' in reference_step["run"]
    assert '"hpa_normal_tissue": os.environ["CI_HPA_NORMAL_TISSUE_SHA256"]' in reference_step["run"]


def test_test_commands_use_bounded_parallelism_with_coverage():
    workflow = yaml.safe_load(Path(".github/workflows/tests.yml").read_text())
    test_steps = workflow["jobs"]["test"]["steps"]
    workflow_command = next(step["run"] for step in test_steps if step["name"] == "Run unit tests")
    local_command = next(
        line for line in Path("test.sh").read_text().splitlines() if "pytest" in line
    )

    for command in (workflow_command, local_command):
        arguments = shlex.split(command)
        assert arguments[arguments.index("-n") + 1] == "2"
        assert arguments[arguments.index("--dist") + 1] == "loadgroup"
        assert arguments[arguments.index("--durations") + 1] == "20"
        assert arguments[arguments.index("--durations-min") + 1] == "0.5"
        assert "--cov=oncoref" in arguments
    assert "--cov-report=xml" in shlex.split(workflow_command)
    assert "--cov-report=term-missing" in shlex.split(local_command)
