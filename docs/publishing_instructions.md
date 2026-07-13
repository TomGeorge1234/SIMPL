# Publishing SIMPL to PyPI

SIMPL uses Hatchling and `hatch-vcs`, so the package version is derived from the
Git tag. Replace `X.Y.Z` below with the release version, for example `0.11.0`.

## 1. Prepare and test the release

Start from a clean checkout of the commit to release:

```bash
git status
uv sync --extra dev
uv run pytest
uv run mkdocs build --strict
```

Commit any final changes before creating the version tag.

## 2. Create the version tag

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
```

Do not push the tag until the artifacts below have built and passed validation.
Without an exact version tag, `hatch-vcs` will produce a development version.

## 3. Build and inspect the package

Delete old artifacts first so a wildcard upload cannot include a previous
release:

```bash
rm -rf dist
uv run python -m build
uv run twine check dist/*
ls -lh dist/
```

This should create one wheel and one source distribution for `X.Y.Z`:

```text
dist/simpl_neuro-X.Y.Z-py3-none-any.whl
dist/simpl_neuro-X.Y.Z.tar.gz
```

Check that both filenames contain the intended stable version before uploading.

## 4. Optionally test on TestPyPI

```bash
uv run twine upload --repository testpypi dist/*
```

Twine will prompt for credentials. Use `__token__` as the username and a
TestPyPI API token as the password. TestPyPI and production PyPI use different
accounts and tokens.

## 5. Push the release and publish to PyPI

```bash
git push origin main
git push origin vX.Y.Z
uv run twine upload dist/*
```

For token authentication, use `__token__` as the username and the production
PyPI API token as the password. Do not put API tokens directly in this file or
commit them to the repository.

PyPI does not allow an uploaded version or artifact to be replaced. If an
incorrect version has already been uploaded, increment the version, create a
new tag, and rebuild from a clean `dist/` directory.

## Command summary

```bash
uv run pytest
uv run mkdocs build --strict
git tag -a vX.Y.Z -m "Release vX.Y.Z"
rm -rf dist
uv run python -m build
uv run twine check dist/*
git push origin main
git push origin vX.Y.Z
uv run twine upload dist/*
```
