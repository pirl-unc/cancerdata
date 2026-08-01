# Publishing the oncodata → oncoref redirect

This directory builds a final `oncodata` 1.6.1 release that depends on and
re-exports `oncoref`, plus a README/warning telling users to migrate. It is a
one-time artifact, separate from the main `oncoref` package.

## Order matters

1. **Publish `oncoref` first** (from the repo root), so the redirect's
   `oncoref>=1.6.0` dependency resolves:

   ```bash
   ./deploy.sh          # lint + test + build + twine upload + tag
   ```

2. **Then publish this redirect:**

   ```bash
   cd pypi-oncodata-redirect
   rm -rf dist && python3 -m build
   python3 -m twine check dist/*
   python3 -m twine upload dist/*
   ```

Both uploads need your PyPI credentials (`~/.pypirc` or a token).

## Notes
- PyPI cannot rename or transfer a project, so `oncoref` is a brand-new
  project and the old `oncodata` 1.4.0 / 1.5.0 releases remain installable.
- This is `oncodata` **1.6.1**. It supersedes the incomplete 1.6.0 redirect,
  which did not preserve legacy submodule imports, base genome support, or the
  `plots` extra.
- After upload you can delete this `pypi-oncodata-redirect/` directory; it is
  not part of the `oncoref` source tree.
