"""Test that the MkDocs documentation builds without errors."""

import subprocess
import sys

import pytest


@pytest.mark.docs
def test_mkdocs_build():
    """mkdocs build --strict must exit cleanly."""
    result = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--clean"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"mkdocs build failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
