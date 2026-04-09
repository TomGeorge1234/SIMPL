"""Tests for simpl.cli."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from simpl.cli import _NOTEBOOK_NAME, _NOTEBOOK_URL, demo, main


class TestDemo:
    def test_downloads_notebook_from_local_source(self, tmp_path, monkeypatch):
        """When run from an editable install, copies from the local source tree."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("builtins.input", lambda _: "n")

        # Create a fake local source notebook
        local_examples = Path(__file__).resolve().parent.parent / "examples"
        assert local_examples.is_dir(), "examples/ dir should exist in source tree"
        local_nb = local_examples / _NOTEBOOK_NAME
        local_exists = local_nb.is_file()

        if local_exists:
            # Real editable install — should copy from source tree
            demo(None)
            dest = tmp_path / _NOTEBOOK_NAME
            assert dest.exists()
            assert dest.stat().st_size > 0

    def test_creates_file_in_cwd(self, tmp_path, monkeypatch):
        """demo() should create the notebook in the current working directory."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("builtins.input", lambda _: "n")
        demo(None)
        assert (tmp_path / _NOTEBOOK_NAME).exists()

    def test_overwrite_aborted(self, tmp_path, monkeypatch):
        """If file exists and user declines, the file should not be modified."""
        monkeypatch.chdir(tmp_path)
        dest = tmp_path / _NOTEBOOK_NAME
        dest.write_text("original content")

        monkeypatch.setattr("builtins.input", lambda _: "n")
        demo(None)
        assert dest.read_text() == "original content"

    def test_overwrite_accepted(self, tmp_path, monkeypatch):
        """If file exists and user says 'y', the file should be replaced."""
        monkeypatch.chdir(tmp_path)
        dest = tmp_path / _NOTEBOOK_NAME
        dest.write_text("original content")

        # First input: overwrite prompt. Second input: launch jupyter prompt.
        inputs = iter(["y", "n"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        demo(None)
        assert dest.read_text() != "original content"

    def test_prompts_to_run_jupyter(self, tmp_path, monkeypatch, capsys):
        """After saving, declining the jupyter prompt should print the command."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("builtins.input", lambda _: "n")
        demo(None)
        captured = capsys.readouterr()
        assert "jupyter notebook" in captured.out

    def test_falls_back_to_download(self, tmp_path, monkeypatch):
        """When no local source tree exists, should attempt to download."""
        monkeypatch.chdir(tmp_path)

        # Patch the local path check to return False
        original_is_file = Path.is_file

        def fake_is_file(self):
            if _NOTEBOOK_NAME in str(self) and "examples" in str(self):
                return False
            return original_is_file(self)

        # Mock urlretrieve to avoid actual network call
        def fake_urlretrieve(url, dest, **kwargs):
            Path(dest).write_text('{"cells": []}')

        monkeypatch.setattr(Path, "is_file", fake_is_file)

        import urllib.request

        monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
        # Two prompts: no existing file so just the jupyter prompt
        monkeypatch.setattr("builtins.input", lambda _: "n")

        demo(None)
        assert (tmp_path / _NOTEBOOK_NAME).exists()

    def test_download_failure_cleans_up(self, tmp_path, monkeypatch):
        """If download fails, partial file should be removed."""
        monkeypatch.chdir(tmp_path)

        original_is_file = Path.is_file

        def fake_is_file(self):
            if _NOTEBOOK_NAME in str(self) and "examples" in str(self):
                return False
            return original_is_file(self)

        def fake_urlretrieve(url, dest, **kwargs):
            Path(dest).write_text("partial")
            raise ConnectionError("network error")

        monkeypatch.setattr(Path, "is_file", fake_is_file)

        import urllib.request

        monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

        with pytest.raises(SystemExit):
            demo(None)

        assert not (tmp_path / _NOTEBOOK_NAME).exists()


class TestMain:
    def test_demo_subcommand(self, tmp_path, monkeypatch):
        """'simpl demo' should call the demo function."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["simpl", "demo"])
        monkeypatch.setattr("builtins.input", lambda _: "n")
        main()
        assert (tmp_path / _NOTEBOOK_NAME).exists()

    def test_no_subcommand_prints_help(self, monkeypatch, capsys):
        """'simpl' with no args should print help."""
        monkeypatch.setattr("sys.argv", ["simpl"])
        main()
        captured = capsys.readouterr()
        assert "demo" in captured.out

    def test_cli_entrypoint(self):
        """The 'simpl' console script should be importable and callable."""
        result = subprocess.run(
            [sys.executable, "-m", "simpl.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "demo" in result.stdout
