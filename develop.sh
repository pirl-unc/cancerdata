#!/bin/bash
set -o errexit
set -o nounset

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

if command -v uv >/dev/null 2>&1; then
    if [[ ! -x .venv/bin/python ]]; then
        uv venv .venv
    fi
    uv pip install --python .venv/bin/python -e ".[dev]"
else
    if [[ ! -x .venv/bin/python ]]; then
        python3 -m venv .venv
    fi
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -e ".[dev]"
fi

.venv/bin/oncoref version
echo "Development CLI: $repo_dir/.venv/bin/oncoref"
