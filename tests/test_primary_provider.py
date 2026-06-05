from __future__ import annotations

import json
import time
from datetime import date

import pytest

from optionsdesk.data.providers.primary import (
    PrimaryProvider,
    _derive_ws_url,
    _parse_market_data_quote,
)


def _instruments():
    return [
        {"instrumentId": {"symbol": "MERV - XMEV - GGAL - CI"}},
        {"instrumentId": {"symbol": "MERV - XMEV - GGAL - 24hs"}},
        {
            "instrumentId": {"symbol": "MERV - XMEV - GFGC7000JU"},
            "maturityDate": "20991231",
        },
        {
            "instrumentId": {"symbol": "MERV - XMEV - GFGV7000JU"},
            "maturityDate": "20991231",
        },
    ]


def test_derive_websocket_url_from_https_base():
    assert _derive_ws_url("https://api.bull.xoms.com.ar/") == "wss://api.bull.xoms.com.ar"


def test_primary_prefers_24hs_spot_and_normalizes_options():
    provider = PrimaryProvider(
        base_url="https://example.test",
        user="user",
        password="password",
    )

    provider._configure_instruments(_instruments())

    assert provider._spot_symbol == "MERV - XMEV - GGAL - 24hs"
    assert provider._option_symbols == {
        "MERV - XMEV - GFGC7000JU": "GFGC7000JU",
        "MERV - XMEV - GFGV7000JU": "GFGV7000JU",
    }
    assert provider._expiry_calendar == {"JU": date(2099, 12, 31)}


def test_primary_ignores_expired_options():
    provider = PrimaryProvider(
        base_url="https://example.test",
        user="user",
        password="password",
    )

    provider._configure_instruments([
        *_instruments(),
        {
            "instrumentId": {"symbol": "MERV - XMEV - GFGC1000EN"},
            "maturityDate": "20200101",
        },
    ])

    assert "MERV - XMEV - GFGC1000EN" not in provider._option_symbols


def test_primary_market_data_updates_chain_incrementally():
    provider = PrimaryProvider(
        base_url="https://example.test",
        user="user",
        password="password",
    )
    provider._configure_instruments(_instruments())
    provider._connected = True

    provider._on_ws_message(None, json.dumps({
        "type": "Md",
        "instrumentId": {"symbol": "MERV - XMEV - GGAL - 24hs"},
        "marketData": {
            "BI": [{"price": 6990, "size": 10}],
            "OF": [{"price": 7010, "size": 20}],
            "LA": {"price": 7000, "size": 5},
            "NV": {"size": 12345},
        },
    }))
    provider._on_ws_message(None, json.dumps({
        "type": "Md",
        "instrumentId": {"symbol": "MERV - XMEV - GFGC7000JU"},
        "marketData": {
            "BI": [{"price": 140, "size": 30}],
            "OF": [{"price": 160, "size": 40}],
            "LA": {"price": 150, "size": 2},
            "NV": {"size": 120},
        },
    }))
    provider._on_ws_message(None, json.dumps({
        "type": "Md",
        "instrumentId": {"symbol": "MERV - XMEV - GFGC7000JU"},
        "marketData": {"OF": [{"price": 155, "size": 25}]},
    }))

    chain = provider.get_options_chain()

    assert chain is not None
    assert chain.spot.symbol == "GGAL"
    assert chain.spot.mid == pytest.approx(7000)
    assert chain.expiry_calendar == {"JU": date(2099, 12, 31)}
    assert chain.options["GFGC7000JU"].bid == pytest.approx(140)
    assert chain.options["GFGC7000JU"].ask == pytest.approx(155)
    assert chain.options["GFGC7000JU"].last == pytest.approx(150)
    assert chain.options["GFGC7000JU"].ask_size == pytest.approx(25)
    assert provider.get_health().source == "PRIMARY"


def test_primary_discovers_and_streams_stock_universe_quotes(monkeypatch):
    from optionsdesk.data.providers import primary

    monkeypatch.setattr(primary.settings, "stock_universe", "GGAL,YPFD")
    provider = PrimaryProvider(
        base_url="https://example.test",
        user="user",
        password="password",
    )
    provider._configure_instruments([
        *_instruments(),
        {"instrumentId": {"symbol": "MERV - XMEV - YPFD - 24hs"}},
    ])
    provider._connected = True

    provider._on_ws_message(None, json.dumps({
        "type": "Md",
        "instrumentId": {"symbol": "MERV - XMEV - YPFD - 24hs"},
        "marketData": {
            "BI": [{"price": 29900, "size": 10}],
            "OF": [{"price": 30100, "size": 20}],
            "LA": {"price": 30000, "size": 5},
            "NV": {"size": 123},
        },
    }))

    quotes = provider.get_quotes(["YPFD"])
    assert provider._stock_symbols["YPFD"] == "MERV - XMEV - YPFD - 24hs"
    assert quotes["YPFD"].mid == pytest.approx(30000)


def test_primary_subscription_batches_at_1000_symbols():
    provider = PrimaryProvider(
        base_url="https://example.test",
        user="user",
        password="password",
        spot_symbol="MERV - XMEV - GGAL - 24hs",
    )
    provider._spot_symbol = "MERV - XMEV - GGAL - 24hs"
    provider._option_symbols = {
        f"MERV - XMEV - GFGC{i}JU": f"GFGC{i}JU"
        for i in range(1001)
    }
    ws = _FakeWebSocket()

    provider._send_subscriptions(ws)

    assert len(ws.sent) == 2
    assert len(json.loads(ws.sent[0])["products"]) == 1000
    assert len(json.loads(ws.sent[1])["products"]) == 2


def test_primary_subscription_preserves_discovered_market_id():
    provider = PrimaryProvider(
        base_url="https://example.test",
        user="user",
        password="password",
    )
    provider._spot_symbol = "GGAL"
    provider._option_symbols = {"GFGC7000JU": "GFGC7000JU"}
    provider._market_ids = {"GGAL": "MERV", "GFGC7000JU": "MERV"}
    ws = _FakeWebSocket()

    provider._send_subscriptions(ws)

    assert json.loads(ws.sent[0])["products"] == [
        {"symbol": "GGAL", "marketId": "MERV"},
        {"symbol": "GFGC7000JU", "marketId": "MERV"},
    ]


def test_primary_marks_socket_stale_when_first_message_never_arrives():
    provider = PrimaryProvider(
        base_url="https://example.test",
        user="user",
        password="password",
    )
    provider._connected = True
    provider._connected_monotonic = time.monotonic() - 61

    assert provider.is_stale(threshold_s=60)


def test_parse_market_data_quote_uses_close_as_last_fallback():
    quote = _parse_market_data_quote(
        "GFGC7000JU",
        {"CL": {"price": 99.5}},
    )

    assert quote.last == pytest.approx(99.5)


def test_parse_market_data_quote_clears_explicitly_empty_book_side():
    previous = _parse_market_data_quote(
        "GFGC7000JU",
        {"BI": [{"price": 100, "size": 2}], "OF": [{"price": 110, "size": 3}]},
    )

    quote = _parse_market_data_quote("GFGC7000JU", {"BI": []}, previous)

    assert quote.bid == 0
    assert quote.bid_size == 0
    assert quote.ask == pytest.approx(110)


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.headers = {}
        self.post_calls = []
        self.get_calls = []

    def post(self, url, headers=None, timeout=10):
        self.post_calls.append((url, headers, timeout))
        return _FakeResponse(200, headers={"X-Auth-Token": "token-123"})

    def get(self, url, headers=None, timeout=10):
        self.get_calls.append((url, headers, timeout))
        return _FakeResponse(200, {"status": "OK", "instruments": _instruments()})


class _FakeWebSocket:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.sent = []

    def run_forever(self, **kwargs):
        self.kwargs["run_forever"] = kwargs
        self.kwargs["on_open"](self)

    def send(self, message):
        self.sent.append(message)

    def close(self):
        pass


def test_primary_connect_authenticates_discovers_and_subscribes():
    sockets = []

    def factory(*args, **kwargs):
        ws = _FakeWebSocket(*args, **kwargs)
        sockets.append(ws)
        return ws

    provider = PrimaryProvider(
        base_url="https://api.example.test",
        user="alice",
        password="secret",
        websocket_factory=factory,
    )
    provider._session = _FakeSession()

    provider.connect()
    try:
        assert provider.is_connected()
        assert provider._session.post_calls[0][0] == "https://api.example.test/auth/getToken"
        assert provider._session.post_calls[0][1] == {
            "X-Username": "alice",
            "X-Password": "secret",
        }
        assert provider._session.get_calls[0][0] == (
            "https://api.example.test/rest/instruments/details"
        )
        assert sockets[0].args[0] == "wss://api.example.test"
        assert sockets[0].kwargs["header"] == ["X-Auth-Token: token-123"]
        subscription = json.loads(sockets[0].sent[0])
        assert subscription["type"] == "smd"
        assert subscription["depth"] == 1
        assert {"symbol": "MERV - XMEV - GGAL - 24hs", "marketId": "ROFX"} in (
            subscription["products"]
        )
    finally:
        provider.disconnect()


def test_primary_reconnect_reuses_same_day_token_and_discovery():
    sockets = []

    def factory(*args, **kwargs):
        ws = _FakeWebSocket(*args, **kwargs)
        sockets.append(ws)
        return ws

    provider = PrimaryProvider(
        base_url="https://api.example.test",
        user="alice",
        password="secret",
        websocket_factory=factory,
    )
    provider._session = _FakeSession()

    provider.connect()
    try:
        provider.reconnect()

        assert len(provider._session.post_calls) == 1
        assert len(provider._session.get_calls) == 1
        assert len(sockets) == 2
    finally:
        provider.disconnect()
