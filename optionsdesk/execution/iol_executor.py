"""Executor real contra la API de operatoria de InvertirOnline (IOL).

A diferencia de PaperExecutor, este SI envia ordenes reales al broker.
Reusa el _TokenManager del IOLProvider (mismo bearer token) y replica el
manejo de 401/re-login y reintentos del lado de lectura.

Endpoints (confirmados contra la doc oficial y un wrapper de referencia):
- POST   /api/v2/operar/Comprar
- POST   /api/v2/operar/Vender
- GET    /api/v2/operaciones/
- GET    /api/v2/operaciones/{numero}
- DELETE /api/v2/operaciones/{numero}
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import requests

from optionsdesk.data.providers.iol import _BASE_URL, _TokenManager
from optionsdesk.execution.base import Order, OrderExecutor, OrderSide, OrderStatus

logger = logging.getLogger(__name__)

_MARKET = "bCBA"
_POST_RETRIES = 1
_RETRY_BASE_SLEEP_S = 0.30

# Estados que devuelve IOL en /operaciones, normalizados a minuscula.
# El set exacto se valida en la primera corrida real; este mapeo cubre los
# nombres habituales de la API v2.
_TERMINAL_FILLED = {"terminada", "terminadaparcial", "ejecutada", "cumplida"}
_TERMINAL_CANCELLED = {"cancelada", "anulada"}
_TERMINAL_REJECTED = {"rechazada", "denegada", "error"}
_LIVE_STATES = {
    "pendiente",
    "enprocesovalidacion",
    "enproceso",
    "revisar",
    "porcompletar",
}


def _map_iol_estado(estado: str) -> OrderStatus:
    key = str(estado or "").strip().lower().replace(" ", "")
    if key in _TERMINAL_FILLED:
        return OrderStatus.FILLED
    if key in _TERMINAL_CANCELLED:
        return OrderStatus.CANCELLED
    if key in _TERMINAL_REJECTED:
        return OrderStatus.REJECTED
    return OrderStatus.SUBMITTED


class IOLExecutor(OrderExecutor):
    """Envia ordenes reales de opciones GGAL via API REST de IOL."""

    def __init__(
        self,
        token_mgr: _TokenManager,
        session: Optional[requests.Session] = None,
        *,
        market: str = _MARKET,
    ) -> None:
        self._token_mgr = token_mgr
        self._session = session or requests.Session()
        self._market = market

    @classmethod
    def from_settings(cls) -> "IOLExecutor":
        """Crea un executor con su propio token IOL (independiente del feed de datos)."""
        from optionsdesk.config.settings import settings

        if not settings.is_iol_configured():
            raise RuntimeError("Faltan IOL_USER e IOL_PASSWORD en .env")
        return cls(_TokenManager(settings.iol_user, settings.iol_password))

    # ── Operatoria ────────────────────────────────────────────────────────────
    def submit(self, order: Order) -> Order:
        path = "operar/Comprar" if order.side == OrderSide.BUY else "operar/Vender"
        body = {
            "mercado": self._market,
            "simbolo": order.symbol,
            "cantidad": int(order.quantity),
            "precio": float(order.limit_price),
            "validez": (order.valid_until or _default_validez()).isoformat(),
            "plazo": order.term,
        }
        data, http_status = self._request("POST", path, json=body)

        numero = _extract_operation_number(data)
        messages = _extract_messages(data)
        if numero is not None:
            order.broker_id = str(numero)
            order.status = OrderStatus.SUBMITTED
            order.notes = messages or "Orden enviada"
        else:
            order.status = OrderStatus.REJECTED
            order.notes = messages or f"IOL rechazo la orden (HTTP {http_status})"
        logger.info(
            "[IOL] %s %s x%d @ %.2f plazo=%s -> %s (%s)",
            order.side, order.symbol, order.quantity, order.limit_price,
            order.term, order.status, order.broker_id or order.notes,
        )
        return order

    def cancel(self, order: Order) -> Order:
        if not order.broker_id:
            order.notes = "Sin numero de operacion para cancelar"
            return order
        data, http_status = self._request("DELETE", f"operaciones/{order.broker_id}")
        if http_status and http_status < 400:
            order.status = OrderStatus.CANCELLED
            order.notes = _extract_messages(data) or "Orden cancelada"
        else:
            order.notes = _extract_messages(data) or f"No se pudo cancelar (HTTP {http_status})"
        return order

    def get_open_orders(self) -> list[Order]:
        data, _ = self._request("GET", "operaciones/")
        if not isinstance(data, list):
            return []
        orders: list[Order] = []
        for item in data:
            order = _parse_operation(item)
            if order is not None and order.status == OrderStatus.SUBMITTED:
                orders.append(order)
        return orders

    def get_order_status(self, numero: str) -> Optional[Order]:
        data, _ = self._request("GET", f"operaciones/{numero}")
        if isinstance(data, dict):
            return _parse_operation(data)
        return None

    def get_history(self, days: int = 7) -> list[dict]:
        """Historial de operaciones (todos los estados) de los ultimos `days` dias."""
        from datetime import timedelta

        hasta = datetime.now().date()
        desde = hasta - timedelta(days=max(days, 0))
        params = {
            "filtro.estado": "todas",
            "filtro.fechaDesde": desde.isoformat(),
            "filtro.fechaHasta": hasta.isoformat(),
        }
        data, _ = self._request("GET", "operaciones/", params=params)
        return data if isinstance(data, list) else []

    def get_positions(self, country: str = "argentina") -> list[dict]:
        """Tenencia actual de opciones GGAL y acciones del universo (para monitoreo real)."""
        data, _ = self._request("GET", f"portafolio/{country}")
        if not isinstance(data, dict):
            return []
            
        from optionsdesk.config.settings import settings
        universe = set(s.strip().upper() for s in settings.stock_universe.split(",") if s.strip())
        
        out: list[dict] = []
        for activo in data.get("activos") or []:
            titulo = activo.get("titulo") or {}
            simbolo = str(titulo.get("simbolo") or "").upper()
            tipo = str(titulo.get("tipo") or "").lower()
            if not (simbolo.startswith("GFG") or "opc" in tipo or simbolo in universe):
                continue
            out.append({
                "simbolo": simbolo,
                "cantidad": _to_float(activo.get("cantidad")),
                "ppc": _to_float(activo.get("ppc") or activo.get("precioPromedio")),
                "ultimo": _to_float(activo.get("ultimoPrecio")),
                "ganancia_pct": _to_float(activo.get("gananciaPorcentaje")),
                "valorizado": _to_float(activo.get("valorizado")),
            })
        return out

    # ── HTTP ──────────────────────────────────────────────────────────────────
    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> tuple[Optional[dict | list], Optional[int]]:
        url = f"{_BASE_URL}/{path}"
        for attempt in range(_POST_RETRIES + 1):
            try:
                headers = self._token_mgr.get_headers()
                resp = self._session.request(
                    method, url, headers=headers, json=json, params=params, timeout=10
                )
                if resp.status_code == 401:
                    logger.warning("IOL operar: 401 en %s, re-login", path)
                    self._token_mgr._do_login()
                    headers = self._token_mgr.get_headers()
                    resp = self._session.request(
                        method, url, headers=headers, json=json, params=params, timeout=10
                    )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < _POST_RETRIES:
                    time.sleep(_RETRY_BASE_SLEEP_S * (attempt + 1))
                    continue
                return _safe_json(resp), resp.status_code
            except requests.RequestException as exc:
                logger.warning("IOL operar %s %s fallo: %s", method, path, exc)
                if attempt < _POST_RETRIES:
                    time.sleep(_RETRY_BASE_SLEEP_S * (attempt + 1))
                    continue
                return {"messages": [str(exc)]}, None
        return None, None


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _default_validez() -> datetime:
    """Validez por defecto: fin de la rueda de hoy (17:30 local)."""
    now = datetime.now().astimezone()
    return now.replace(hour=17, minute=30, second=0, microsecond=0)


def _safe_json(resp: requests.Response) -> Optional[dict | list]:
    try:
        return resp.json()
    except ValueError:
        return None


def _extract_operation_number(data) -> Optional[int]:
    if not isinstance(data, dict):
        return None
    for key in ("numeroOperacion", "numero", "idOperacion"):
        val = data.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return None


def _extract_messages(data) -> str:
    if not isinstance(data, dict):
        return ""
    raw = data.get("messages") or data.get("mensajes") or data.get("message")
    if isinstance(raw, list):
        parts: list[str] = []
        for m in raw:
            if isinstance(m, dict):
                parts.append(str(m.get("title") or m.get("description") or m))
            else:
                parts.append(str(m))
        return " | ".join(p for p in parts if p)
    return str(raw) if raw else ""


def _parse_operation(item: dict) -> Optional[Order]:
    if not isinstance(item, dict):
        return None
    tipo = str(item.get("tipo") or "").strip().lower()
    side = OrderSide.SELL if tipo.startswith("vent") else OrderSide.BUY
    try:
        order = Order(
            symbol=str(item.get("simbolo") or ""),
            side=side,
            quantity=int(item.get("cantidad") or 0),
            limit_price=float(item.get("precio") or 0.0),
            strategy_ref="IOL",
            term=str(item.get("plazo") or "t1"),
        )
    except (TypeError, ValueError):
        return None
    order.broker_id = str(item.get("numero") or item.get("numeroOperacion") or "") or None
    order.status = _map_iol_estado(item.get("estado") or item.get("estadoActual") or "")
    return order
