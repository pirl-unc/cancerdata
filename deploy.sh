#!/usr/bin/env bash
set -euo pipefail

REPO="${GITHUB_REPOSITORY:-pirl-unc/oncoref}"
PYTHON_COMMAND="${PYTHON:-python}"
if ! PYTHON_BIN="$(command -v "$PYTHON_COMMAND")"; then
    echo "Python interpreter not found: $PYTHON_COMMAND" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c \
    'import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)'
then
    echo "deploy.sh requires a virtualenv Python; activate one or set PYTHON" >&2
    exit 1
fi

# Keep lint.sh and test.sh on the same interpreter used to build and upload.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

VERSION="$("$PYTHON_BIN" -c 'from oncoref.version import __version__; print(__version__)')"
TAG="v${VERSION}"
PYPI_URL="https://pypi.org/project/oncoref/${VERSION}/"
PYPI_PUBLISHED_MARKER="<!-- oncoref-deploy-pypi-published -->"

./test.sh
"$PYTHON_BIN" -m pip install --upgrade build twine
rm -rf dist
"$PYTHON_BIN" -m build
"$PYTHON_BIN" -m twine upload dist/*
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "tag $TAG already exists locally"
else
    git tag "$TAG"
fi
git push --tags

RELEASE_NOTES="${PYPI_PUBLISHED_MARKER}

PyPI package: ${PYPI_URL}

This release was published by \`./deploy.sh\` after local lint and test checks passed."

if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    gh release edit "$TAG" \
        --repo "$REPO" \
        --title "$TAG" \
        --notes "$RELEASE_NOTES"
else
    gh release create "$TAG" \
        --repo "$REPO" \
        --verify-tag \
        --title "$TAG" \
        --notes "$RELEASE_NOTES" \
        --generate-notes
fi
