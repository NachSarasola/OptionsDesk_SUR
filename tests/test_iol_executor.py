from __future__ import annotations

from datetime import datetime

import pytest

from optionsdesk.execution.base import Order, OrderSide, OrderStatus
from optionsdesk.execution.iol_executor import IOLExecutor, _map_iol_estado


class _DummyTokenMgr:
    def get_headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer x"}

    def _do_login(self) -> None:  # pragma: no cover - no se ejercita aca
        pass


class _DummyResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class _RecordingSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def request(self, method, url, headers=None, json=None, params=None, timeout=10):  # noqa: ARG002
        self.calls.append({"method": method, "url": url, "json": json, "params": params})
        return self._responses.pop(0)


def _executor(responses):
    session = _RecordingSession(responses)
    return IOLExecutor(_DummyTokenMgr(), session=session), session


def test_submit_buy_posts_correct_url_and_body():
    ex, session = _executor([_DummyResponse(200, {"numeroOperacion": 9876})])
    order = Order(
        symbol="GFGC7000F",
        side=OrderSide.BUY,
        quantity=5,
        limit_price=123.5,
        strategy_ref="MANUAL",
        term="t1",
        valid_until=datetime(2026, 6, 3, 17, 30, 0),
    )

    out = ex.submit(order)

    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/operar/Comprar")
    assert call["json"] == {
        "mercado": "bCBA",
        "simbolo": "GFGC7000F",
        "cantidad": 5,
        "precio": 123.5,
        "validez": "2026-06-03T17:30:00",
        "plazo": "t1",
    }
    assert out.status == OrderStatus.SUBMITTED
    assert out.broker_id == "9876"


def test_submit_sell_uses_vender_endpoint():
    ex, session = _executor([_DummyResponse(200, {"numeroOperacion": 1})])
    order = Order(
        symbol="GFGV7000F",
        side=OrderSide.SELL,
        quantity=2,
        limit_price=80.0,
        strategy_ref="MANUAL",
        term="t0",
        valid_until=datetime(2026, 6, 3, 17, 30, 0),
    )

    ex.submit(order)

    call = session.calls[0]
    assert call["url"].endswith("/operar/Vender")
    assert call["json"]["plazo"] == "t0"


def test_submit_rejected_when_no_operation_number():
    ex, _ = _executor(
        [_DummyResponse(400, {"messages": [{"title": "Saldo insuficiente"}]})]
    )
    order = Order(
        symbol="GFGC7000F",
        side=OrderSide.BUY,
        quantity=1,
        limit_price=100.0,
        strategy_ref="MANUAL",
    )

    out = ex.submit(order)

    assert out.status == OrderStatus.REJECTED
    assert "Saldo insuficiente" in out.notes
    assert out.broker_id is None


def test_cancel_sends_delete_to_operaciones():
    ex, session = _executor([_DummyResponse(200, {"messages": ["ok"]})])
    order = Order(
        symbol="GFGC7000F",
        side=OrderSide.BUY,
        quantity=1,
        limit_price=100.0,
        strategy_ref="MANUAL",
    )
    order.broker_id = "4321"

    out = ex.cancel(order)

    call = session.calls[0]
    assert call["method"] == "DELETE"
    assert call["url"].endswith("/operaciones/4321")
    assert out.status == OrderStatus.CANCELLED


def test_get_open_orders_filters_live_states():
    payload = [
        {"numero": 1, "tipo": "Compra", "estado": "Pendiente", "simbolo": "GFGC7000F",
         "cantidad": 3, "precio": 120.0, "plazo": "t1"},
        {"numero": 2, "tipo": "Venta", "estado": "Terminada", "simbolo": "GFGV7000F",
         "cantidad": 1, "precio": 80.0, "plazo": "t1"},
    ]
    ex, _ = _executor([_DummyResponse(200, payload)])

    orders = ex.get_open_orders()

    assert len(orders) == 1
    assert orders[0].broker_id == "1"
    assert orders[0].side == OrderSide.BUY
    assert orders[0].status == OrderStatus.SUBMITTED


@pytest.mark.parametrize(
    "estado,expected",
    [
        ("Terminada", OrderStatus.FILLED),
        ("Cancelada", OrderStatus.CANCELLED),
        ("Rechazada", OrderStatus.REJECTED),
        ("Pendiente", OrderStatus.SUBMITTED),
        ("en proceso", OrderStatus.SUBMITTED),
    ],
)
def test_map_iol_estado(estado, expected):
    assert _map_iol_estado(estado) == expected
