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
    option = {
        "simbolo": "GFGC4000JU",
        "fechaVencimiento": "2026-06-19T00:00:00",
    }
    quote = {
        "ultimoPrecio": 150.0,
        "puntas": [{"precioCompra": 149.0, "precioVenta": 151.0}],
        "volumenNominal": 100,
        "fechaHora": "2026-06-01T15:00:00-03:00",
    }

    def fake_get(path, quiet_statuses=None):  # noqa: ARG001
        if path.endswith("/GGAL/Cotizacion"):
            return spot
        if path.endswith("/GGAL/Opciones"):
            return [option]
        if path.endswith("/GFGC4000JU/Cotizacion"):
            return quote
        raise AssertionError(path)

    monkeypatch.setattr(p, "_get", fake_get)

    chain = p.get_options_chain()

    assert chain is not None
    assert chain.expiry_calendar == {"JU": date(2026, 6, 19)}
    assert "GFGC4000JU" in chain.options
