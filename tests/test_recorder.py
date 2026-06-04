"""Tests del arranque explicito del recorder."""
from __future__ import annotations

from datetime import datetime

import pytest

from optionsdesk.data import recorder
from optionsdesk.data.providers.base import MarketDataHealth, Quote


def test_build_provider_requires_real_credentials(monkeypatch):
    monkeypatch.setattr(recorder.settings, "primary_base_url", "")
    monkeypatch.setattr(recorder.settings, "primary_user", "")
    monkeypatch.setattr(recorder.settings, "primary_password", "")
    monkeypatch.setattr(recorder.settings, "iol_user", "")
    monkeypatch.setattr(recorder.settings, "iol_password", "")
    monkeypatch.setattr(recorder.settings, "hb_dni", "")
    monkeypatch.setattr(recorder.settings, "hb_user", "")
    monkeypatch.setattr(recorder.settings, "hb_password", "")
    monkeypatch.setattr(recorder.settings, "hb_broker_id", 0)

    with pytest.raises(RuntimeError, match="Configura credenciales"):
        recorder._build_provider()


def test_build_provider_prefers_primary(monkeypatch):
    from optionsdesk.data.providers import primary

    class FakeProvider:
        connected = False

        def connect(self):
            self.connected = True

    fake = FakeProvider()
    monkeypatch.setattr(recorder.settings, "primary_base_url", "https://example.test")
    monkeypatch.setattr(recorder.settings, "primary_user", "user")
    monkeypatch.setattr(recorder.settings, "primary_password", "password")
    monkeypatch.setattr(recorder.settings, "iol_user", "iol-user")
    monkeypatch.setattr(recorder.settings, "iol_password", "iol-password")
    monkeypatch.setattr(primary, "PrimaryProvider", lambda: fake)

    assert recorder._build_provider() is fake
    assert fake.connected


def test_market_session_gate_excludes_after_hours_and_weekends():
    assert recorder._is_byma_market_session(datetime(2026, 6, 2, 11, 0))
    assert not recorder._is_byma_market_session(datetime(2026, 6, 2, 17, 0))
    assert not recorder._is_byma_market_session(datetime(2026, 6, 7, 12, 0))


def test_recorder_does_not_reconnect_stale_feed_after_hours(monkeypatch):
    class FakeProvider:
        reconnect_calls = 0

        def is_stale(self, threshold_s=60):  # noqa: ARG002
            return True

        def reconnect(self):
            self.reconnect_calls += 1

        def is_connected(self):
            return True

        def get_options_chain(self):
            return None

    provider = FakeProvider()
    monkeypatch.setattr(recorder, "_is_byma_market_session", lambda: False)
    monkeypatch.setattr(recorder.ChainRecorder, "_write_status", lambda *args, **kwargs: None)

    assert not recorder.ChainRecorder(provider).take_snapshot()
    assert provider.reconnect_calls == 0


def test_recorder_reconnects_stale_feed_during_market_session(monkeypatch):
    class FakeProvider:
        reconnect_calls = 0

        def is_stale(self, threshold_s=60):  # noqa: ARG002
            return True

        def reconnect(self):
            self.reconnect_calls += 1

        def is_connected(self):
            return True

        def get_options_chain(self):
            return None

    provider = FakeProvider()
    monkeypatch.setattr(recorder, "_is_byma_market_session", lambda: True)
    monkeypatch.setattr(recorder.ChainRecorder, "_write_status", lambda *args, **kwargs: None)

    assert not recorder.ChainRecorder(provider).take_snapshot()
    assert provider.reconnect_calls == 1


def test_recorder_keeps_running_when_generic_reconnect_fails(monkeypatch):
    class FakeProvider:
        def is_stale(self, threshold_s=60):  # noqa: ARG002
            return True

        def reconnect(self):
            raise RuntimeError("feed down")

        def get_options_chain(self):
            return None

    monkeypatch.setattr(recorder, "_is_byma_market_session", lambda: True)
    monkeypatch.setattr(recorder.ChainRecorder, "_write_status", lambda *args, **kwargs: None)

    assert not recorder.ChainRecorder(FakeProvider()).take_snapshot()


def test_recorder_applies_conservative_floor_for_iol(monkeypatch):
    class FakeProvider:
        def get_health(self):
            return MarketDataHealth(source="IOL", connected=True)

    monkeypatch.setattr(recorder.settings, "recorder_interval_s", 15)
    monkeypatch.setattr(recorder.settings, "iol_recorder_interval_s", 900)

    assert recorder.ChainRecorder(FakeProvider())._interval == 900


def test_recorder_persists_lightweight_spot_between_chain_snapshots(monkeypatch):
    saved = []

    class FakeProvider:
        def get_health(self):
            return MarketDataHealth(source="IOL", connected=True)

        def get_spot(self):
            return Quote("GGAL", 99.0, 101.0, 100.0, 1.0, datetime.now())

    monkeypatch.setattr(recorder, "_is_byma_market_session", lambda: True)
    monkeypatch.setattr(recorder, "append_spot_tape", lambda spot, source: saved.append((spot, source)))

    assert recorder.ChainRecorder(FakeProvider())._record_spot_sample()
    assert saved == [(100.0, "IOL")]
