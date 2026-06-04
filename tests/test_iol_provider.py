from __future__ import annotations

from datetime import date, datetime

import pytest

from optionsdesk.data.providers.iol import IOLProvider, _parse_quote


def test_parse_quote_with_nested_cotizacion_and_puntas():
    payload = {
        "cotizacion": {
            "puntas": [
                {
                    "precioCompra": 120.5,
                    "precioVenta": 121.5,
                    "cantidadCompra": 10,
                    "cantidadVenta": 20,
                }
            ],
            "ultimo": {"precio": 121.0, "fecha": "2026-05-27T15:30:00Z"},
            "volumenNominal": 12345,
        }
    }
    q = _parse_quote("GFGC7000F", payload)
    assert q is not None
    assert q.bid == pytest.approx(120.5)
    assert q.ask == pytest.approx(121.5)
    assert q.last == pytest.approx(121.0)
    assert q.volume == pytest.approx(12345.0)
    assert q.bid_size == pytest.approx(10)
    assert q.ask_size == pytest.approx(20)
    assert isinstance(q.timestamp, datetime)


def test_parse_quote_with_flat_payload_and_ultimo_precio():
    payload = {
        "cantidadOperaciones": 2,
        "ultimoPrecio": 55.0,
        "volumeNominal": 400.0,
        "fechaHora": "2026-05-27T15:31:00Z",
        "puntas": [],
    }
    q = _parse_quote("GFGV7000F", payload)
    assert q is not None
    assert q.last == pytest.approx(55.0)
    assert q.volume == pytest.approx(400.0)


def test_parse_quote_fallback_to_cierre_anterior_when_no_last():
    payload = {
        "cantidadOperaciones": 0,
        "cierreAnterior": 120.0,
        "puntas": [],
    }
    q = _parse_quote("GFGC7000F", payload)
    assert q is not None
    assert q.last == pytest.approx(120.0)


def test_parse_spot_uses_cierre_anterior_when_no_intraday_last():
    payload = {
        "cantidadOperaciones": 0,
        "ultimoPrecio": 0,
        "cierreAnterior": 7127.5,
        "puntas": [],
    }
    q = _parse_quote("GGAL", payload)
    assert q is not None
    assert q.last == pytest.approx(7127.5)


class _DummyTokenMgr:
    def __init__(self):
        self.login_calls = 0

    def get_headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer x"}

    def _do_login(self) -> None:
        self.login_calls += 1


class _DummyResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class _DummySession:
    def __init__(self, responses):
        self._responses = list(responses)

    def get(self, url, headers=None, timeout=10):  # noqa: ARG002
        return self._responses.pop(0)


def test_iol_get_retries_on_5xx(monkeypatch):
    p = IOLProvider()
    p._token_mgr = _DummyTokenMgr()
    p._connected = True
    p._session = _DummySession(
        [
            _DummyResponse(500),
            _DummyResponse(200, {"ok": True}),
        ]
    )
    monkeypatch.setattr("optionsdesk.data.providers.iol.time.sleep", lambda x: None)
    out = p._get("bCBA/Titulos/GGAL/Cotizacion")
    assert out == {"ok": True}
    assert p.get_health().retries >= 1


def test_iol_chain_carries_observed_multiletter_expiry(monkeypatch):
    p = IOLProvider()
    p._connected = True
    p._token_mgr = _DummyTokenMgr()
    spot = {
        "ultimoPrecio": 4000.0,
        "puntas": [{"precioCompra": 3999.0, "precioVenta": 4001.0}],
        "volumenNominal": 1000,
        "fechaHora": "2026-06-01T15:00:00-03:00",
    }
    quote = {
        "ultimoPrecio": 150.0,
        "puntas": [{"precioCompra": 149.0, "precioVenta": 151.0}],
        "volumenNominal": 100,
        "fechaHora": "2026-06-01T15:00:00-03:00",
    }
    option = {
        "simbolo": "GFGC4000JU",
        "fechaVencimiento": "2026-06-19T00:00:00",
        "cotizacion": quote,
    }

    def fake_get(path, quiet_statuses=None):  # noqa: ARG001
        if path.endswith("/GGAL/Cotizacion"):
            return spot
        if path.endswith("/GGAL/Opciones"):
            return [option]
        raise AssertionError(path)

    monkeypatch.setattr(p, "_get", fake_get)
    monkeypatch.setattr("optionsdesk.data.providers.iol.settings.iol_option_quotes_per_cycle", 0)

    chain = p.get_options_chain()

    assert chain is not None
    assert chain.expiry_calendar == {"JU": date(2026, 6, 19)}
    assert "GFGC4000JU" in chain.options


def test_iol_chain_fetches_only_limited_nearest_quotes_when_listing_is_empty(monkeypatch):
    p = IOLProvider()
    p._connected = True
    p._token_mgr = _DummyTokenMgr()
    paths = []
    spot = {
        "ultimoPrecio": 4000.0,
        "puntas": [{"precioCompra": 3999.0, "precioVenta": 4001.0}],
    }
    items = [
        {
            "simbolo": symbol,
            "fechaVencimiento": "2026-06-19T00:00:00",
            "cotizacion": {"cantidadOperaciones": 0, "puntas": []},
        }
        for symbol in ("GFGC4000JU", "GFGV4000JU", "GFGC4500JU")
    ]

    def fake_get(path, quiet_statuses=None):  # noqa: ARG001
        paths.append(path)
        if path.endswith("/GGAL/Cotizacion"):
            return spot
        if path.endswith("/GGAL/Opciones"):
            return items
        return {
            "cantidadOperaciones": 1,
            "ultimoPrecio": 100.0,
            "puntas": [{"precioCompra": 99.0, "precioVenta": 101.0}],
        }

    monkeypatch.setattr(p, "_get", fake_get)
    monkeypatch.setattr("optionsdesk.data.providers.iol.settings.iol_option_quotes_per_cycle", 2)

    chain = p.get_options_chain()

    assert chain is not None
    assert set(chain.options) == {"GFGC4000JU", "GFGV4000JU"}
    assert sum(path.endswith("/Cotizacion") for path in paths) == 3  # spot + 2 opciones


def test_iol_get_quote_uses_requested_stock_symbol(monkeypatch):
    p = IOLProvider()
    p._connected = True
    p._token_mgr = _DummyTokenMgr()
    paths = []

    def fake_get(path, quiet_statuses=None):  # noqa: ARG001
        paths.append(path)
        return {
            "ultimoPrecio": 41200.0,
            "puntas": [{"precioCompra": 41150.0, "precioVenta": 41250.0}],
            "volumenNominal": 1234,
            "fechaHora": "2026-06-03T15:00:00-03:00",
        }

    monkeypatch.setattr(p, "_get", fake_get)

    quote = p.get_quote("YPFD")

    assert quote is not None
    assert quote.symbol == "YPFD"
    assert quote.bid == pytest.approx(41150.0)
    assert quote.ask == pytest.approx(41250.0)
    assert paths == ["bCBA/Titulos/YPFD/Cotizacion"]


def test_iol_get_quotes_returns_multiple_symbols(monkeypatch):
    p = IOLProvider()
    calls = []

    def fake_get_quote(symbol):
        calls.append(symbol)
        return _parse_quote(
            symbol,
            {
                "ultimoPrecio": 100.0,
                "puntas": [{"precioCompra": 99.0, "precioVenta": 101.0}],
            },
        )

    monkeypatch.setattr(p, "get_quote", fake_get_quote)

    quotes = p.get_quotes(["ggal", "YPFD", "GGAL"])

    assert set(quotes) == {"GGAL", "YPFD"}
    assert set(calls) == {"GGAL", "YPFD"}
