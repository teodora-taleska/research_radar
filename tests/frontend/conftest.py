"""Fixtures for the frontend tests."""

import pytest


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """An empty evaluation results folder that the frontend will read."""
    folder = tmp_path / "results"
    folder.mkdir()
    monkeypatch.setenv("RESEARCH_RADAR_RESULTS_DIR", str(folder))
    return folder
