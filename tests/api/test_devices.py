"""Tests for ``speechtotext.api.devices.DeviceRegistry``."""

from __future__ import annotations

from pathlib import Path

import pytest

from speechtotext.api.devices import DeviceRegistry


@pytest.fixture
def registry(tmp_path: Path) -> DeviceRegistry:
    reg = DeviceRegistry(db_path=tmp_path / "devices.db")
    yield reg
    reg.close()


class TestRegister:
    def test_register_then_get(self, registry: DeviceRegistry) -> None:
        registry.register("dev-aaa", "pk-b64", "iPad")
        row = registry.get("dev-aaa")
        assert row is not None
        assert row["device_id"] == "dev-aaa"
        assert row["pubkey_b64"] == "pk-b64"
        assert row["name"] == "iPad"
        assert row["paired_at"]
        assert row["last_seen"] is None

    def test_register_replaces_existing(self, registry: DeviceRegistry) -> None:
        registry.register("dev-aaa", "pk-v1", "iPad")
        registry.register("dev-aaa", "pk-v2", "iPad Renamed")
        row = registry.get("dev-aaa")
        assert row["pubkey_b64"] == "pk-v2"
        assert row["name"] == "iPad Renamed"

    def test_get_missing_returns_none(self, registry: DeviceRegistry) -> None:
        assert registry.get("dev-unknown") is None


class TestLastSeen:
    def test_update_last_seen_sets_timestamp(
        self, registry: DeviceRegistry
    ) -> None:
        registry.register("dev-aaa", "pk", "test")
        assert registry.get("dev-aaa")["last_seen"] is None
        registry.update_last_seen("dev-aaa")
        assert registry.get("dev-aaa")["last_seen"] is not None

    def test_update_last_seen_unknown_device_is_noop(
        self, registry: DeviceRegistry
    ) -> None:
        # Should not raise — silently no-op on missing row.
        registry.update_last_seen("dev-never")

    def test_update_last_seen_throttled_within_window(
        self, registry: DeviceRegistry
    ) -> None:
        # A signed request from a frequently-polling device must not turn
        # a read endpoint into a SQLite write on every call. The first
        # update writes; a second within the throttle window is skipped.
        registry.register("dev-aaa", "pk", "test")
        registry.update_last_seen("dev-aaa")
        first = registry.get("dev-aaa")["last_seen"]
        assert first is not None
        registry.update_last_seen("dev-aaa")
        assert registry.get("dev-aaa")["last_seen"] == first

    def test_update_last_seen_writes_after_interval(
        self, registry: DeviceRegistry, monkeypatch
    ) -> None:
        import time

        import speechtotext.api.devices as devices_mod

        registry.register("dev-aaa", "pk", "test")
        registry.update_last_seen("dev-aaa")
        first = registry.get("dev-aaa")["last_seen"]
        # Collapse the throttle window so the next call writes again.
        monkeypatch.setattr(devices_mod, "LAST_SEEN_MIN_INTERVAL_S", 0.0)
        time.sleep(0.01)
        registry.update_last_seen("dev-aaa")
        assert registry.get("dev-aaa")["last_seen"] != first


class TestList:
    def test_list_orders_by_paired_at_desc(
        self, registry: DeviceRegistry
    ) -> None:
        import time

        registry.register("dev-old", "pk", "old")
        time.sleep(0.01)
        registry.register("dev-new", "pk", "new")
        items = registry.list_all()
        assert [r["device_id"] for r in items] == ["dev-new", "dev-old"]


class TestDelete:
    def test_delete_removes_record(self, registry: DeviceRegistry) -> None:
        registry.register("dev-aaa", "pk", "test")
        registry.delete("dev-aaa")
        assert registry.get("dev-aaa") is None

    def test_delete_missing_is_noop(self, registry: DeviceRegistry) -> None:
        registry.delete("dev-never")


class TestPersistence:
    def test_data_survives_close_and_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "devices.db"
        first = DeviceRegistry(db_path=path)
        first.register("dev-aaa", "pk", "test")
        first.close()
        second = DeviceRegistry(db_path=path)
        try:
            assert second.get("dev-aaa") is not None
        finally:
            second.close()
