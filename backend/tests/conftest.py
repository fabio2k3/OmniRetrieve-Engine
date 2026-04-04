from pathlib import Path
import pytest


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def network() -> bool:
    return False