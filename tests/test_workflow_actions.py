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
