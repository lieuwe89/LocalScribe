from unittest.mock import patch

import pytest

from speechtotext.backend import resolve_backend
from speechtotext.config import Config


@pytest.fixture
def cfg() -> Config:
    return Config(backend="auto")


def test_cli_flag_wins(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STT_BACKEND", "cuda")
    cfg.backend = "cpu"
    assert resolve_backend(cli_flag="mps", config=cfg) == "mps"


def test_env_var_wins_over_config(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STT_BACKEND", "cuda")
    cfg.backend = "cpu"
    with patch("speechtotext.backend._cuda_available", return_value=True):
        assert resolve_backend(cli_flag=None, config=cfg) == "cuda"


def test_config_wins_when_no_flag_no_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("STT_BACKEND", raising=False)
    cfg = Config(backend="cpu")
    assert resolve_backend(cli_flag=None, config=cfg) == "cpu"


def test_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("STT_BACKEND", raising=False)
    cfg = Config(backend="auto")
    with (
        patch("speechtotext.backend._cuda_available", return_value=True),
        patch("speechtotext.backend._mps_available", return_value=True),
    ):
        assert resolve_backend(cli_flag=None, config=cfg) == "cuda"


def test_auto_falls_back_to_mps_then_cpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("STT_BACKEND", raising=False)
    cfg = Config(backend="auto")
    with (
        patch("speechtotext.backend._cuda_available", return_value=False),
        patch("speechtotext.backend._mps_available", return_value=True),
    ):
        assert resolve_backend(cli_flag=None, config=cfg) == "mps"
    with (
        patch("speechtotext.backend._cuda_available", return_value=False),
        patch("speechtotext.backend._mps_available", return_value=False),
    ):
        assert resolve_backend(cli_flag=None, config=cfg) == "cpu"


def test_invalid_cli_flag_rejected(cfg: Config):
    with pytest.raises(ValueError):
        resolve_backend(cli_flag="tpu", config=cfg)
