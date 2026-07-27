from pathlib import Path

import yaml


def test_pages_artifact_name_is_unique_per_run_attempt():
    workflow = yaml.safe_load(Path(".github/workflows/docs.yml").read_text())
    deploy_job = workflow["jobs"]["deploy"]
    artifact_name = "github-pages-${{ github.run_id }}-${{ github.run_attempt }}"
    upload_step = next(
        step
        for step in deploy_job["steps"]
        if step.get("uses", "").startswith("actions/upload-pages-artifact@")
    )
    deploy_step = next(
        step
        for step in deploy_job["steps"]
        if step.get("uses", "").startswith("actions/deploy-pages@")
    )

    assert deploy_job["env"]["PAGES_ARTIFACT_NAME"] == artifact_name
    assert upload_step["with"]["name"] == "${{ env.PAGES_ARTIFACT_NAME }}"
    assert deploy_step["with"]["artifact_name"] == "${{ env.PAGES_ARTIFACT_NAME }}"
