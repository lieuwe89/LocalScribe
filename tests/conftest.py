from pathlib import Path
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def audio_fixture_dir(fixture_dir: Path) -> Path:
    return fixture_dir / "audio"
