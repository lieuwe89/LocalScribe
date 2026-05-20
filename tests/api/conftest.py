import pytest
import pytest_asyncio  # noqa: F401  # ensure plugin loads


@pytest.fixture(autouse=True)
def _isolated_library_db(tmp_path_factory, monkeypatch):
    """Point every create_app() in api tests at a throw-away SQLite DB so
    the suite doesn't trample the developer's real ~/Library/.../library.db
    and tests can't leak state across each other.

    Implementation note: monkeypatch.setattr by string path is brittle
    across tests that wipe sys.modules (e.g. test_sidecar_cold_start),
    because Python's import system does not re-bind a child attribute on
    a parent package when the child is already in sys.modules. Resolving
    the module via `import ... as` and monkeypatching the live object
    directly side-steps the broken parent attribute walk.
    """
    import speechtotext.api.library_db as _library_db
    db_dir = tmp_path_factory.mktemp("libdb")
    monkeypatch.setattr(
        _library_db, "default_db_path",
        lambda: db_dir / "library.db",
    )


def pytest_collection_modifyitems(config, items):
    pass
