"""Root conftest.py — shared fixtures for all CodeVerify tests."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load defaults from .env.example so tests never rely on the host environment.

    Individual tests can still override via monkeypatch.setenv().
    Only sets a variable if it is not already set in the real environment.
    """
    env_file = Path(__file__).parent / ".env.example"
    if not env_file.exists():
        return

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not os.environ.get(key):
            monkeypatch.setenv(key, value)
