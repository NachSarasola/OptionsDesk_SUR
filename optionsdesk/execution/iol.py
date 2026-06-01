"""Executor real via la API REST de InvertirOnLine (IOL).

ADVERTENCIA: este executor envía órdenes REALES a IOL con dinero real.
Usarlo solo después de validar la estrategia con PaperExecutor.

Mapeo Order → IOL:
  OrderSide.BUY  → POST /api/v2/operar/Comprar
  OrderSide.SELL → POST /api/v2/operar/Vender

Plazo de liquidación:
  Opciones (símbolo empieza con GFG) → "t1" (T+1, estándar BYMA opciones)
  Acciones (GGAL)                    → "t2" (T+2, estándar BYMA acciones)

El campo `broker_id` del Order se rellena con el `numeroPedido` que
devuelve IOL al enviar la orden.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

from optionsdesk.execution.base import Order, OrderExecutor, OrderSide, OrderStatus

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.invertironline.com/api/v2"
_MARKET   = "bCBA"


class IOLExecutor(OrderExecutor):
    """Executor de órdenes reales vía IOL.

    Requiere una instancia de `_TokenManager` del IOLProvider, que gestiona
    la autenticación compartida entre provider y executor.

    Uso típico:
        from optionsdesk.data.providers.iol import IOLProvider
        from optionsdesk.execution.iol import IOLExecutor

        provider = IOLProvider()
        provider.connect()
        executor = IOLExecutor(provider._token_mgr)
        order = executor.submit(my_order)
    """

    def __init__(self, token_manager) -> None:  # type: ignore[annotation-unchecked]
        """
        Args:
            token_manager: instancia de `_TokenManager` del IOLProvider conectado.
        """
        self._token_mgr = token_manager
        self._session = requests.Session()
        self._local_orders: list[Order] = []

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def submit(self, order: Order) -> Order:
        """Envía la orden a IOL y actualiza el estado.

        Si IOL devuelve un `numeroPedido`, se guarda en `order.broker_id`.
        En caso de error HTTP, el estado queda en REJECTED con nota.
        """
        endpoint = "operar/Comprar" if order.side == OrderSide.BUY else "operar/Vender"
        body = self._build_body(order)

        try:
            resp = self._post(endpoint, body)
            if resp is None:
                raise RuntimeError("Sin respuesta del servidor.")

            numero = (
                resp.get("numeroPedido")
                or resp.get("numero")
                or resp.get("id")
            )
            order.status    = OrderStatus.SUBMITTED
            order.broker_id = str(numero) if numero is not None else "IOL-OK"
            self._local_orders.append(order)
            logger.info(
                "[IOL] %s %s x%d @ %.2f  pedido=%s  ref=%s",
                order.side, order.symbol, order.quantity,
                order.limit_price, order.broker_id, order.strategy_ref,
            )

        except requests.HTTPError as exc:
            order.status = OrderStatus.REJECTED
            order.notes  = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            logger.error("[IOL] Orden rechazada: %s", order.notes)

        except Exception as exc:
            order.status = OrderStatus.REJECTED
            order.notes  = str(exc)
            logger.error("[IOL] Error al enviar orden: %s", exc)

        return order

    def get_open_orders(self) -> list[Order]:
        """Consulta las órdenes vigentes en IOL y las reconcilia con locales."""
        data = self._get("operaciones")
        if not isinstance(data, list):
            # Fallback: devolver las locales en estado SUBMITTED
            return [o for o in self._local_orders if o.status == OrderStatus.SUBMITTED]

        # Construir set de pedidos vigentes según IOL para sincronizar estado local
        iol_open: set[str] = set()
        for item in data:
            estado = str(item.get("estado", "")).lower()
            # IOL usa "EnVigor", "Parcial", "Pendiente" para órdenes activas
            if any(s in estado for s in ("envigor", "parcial", "pendiente")):
                num = item.get("numeroPedido") or item.get("numero")
                if num:
                    iol_open.add(str(num))

        # Marcar como FILLED/CANCELLED las que ya no están en IOL
        for o in self._local_orders:
            if o.status == OrderStatus.SUBMITTED and o.broker_id not in iol_open:
                o.status = OrderStatus.FILLED
                logger.debug("[IOL] Orden %s marcada FILLED (ya no está en vigentes).", o.broker_id)

        return [o for o in self._local_orders if o.status == OrderStatus.SUBMITTED]

    def cancel(self, order: Order) -> Order:
        """Cancela la orden en IOL via DELETE /api/v2/operaciones/{numeroPedido}."""
        if not order.broker_id:
            order.status = OrderStatus.CANCELLED
            order.notes  = "Sin broker_id — cancelación solo local."
            return order

        try:
            resp = self._delete(f"operaciones/{order.broker_id}")
            order.status = OrderStatus.CANCELLED
            logger.info("[IOL] Orden %s cancelada.", order.broker_id)
        except requests.HTTPError as exc:
            order.notes = f"Cancel falló: HTTP {exc.response.status_code}"
            logger.warning("[IOL] %s", order.notes)
        except Exception as exc:
            order.notes = f"Cancel falló: {exc}"
            logger.warning("[IOL] %s", order.notes)

        return order

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _build_body(self, order: Order) -> dict:
        """Construye el body JSON para la API de IOL."""
        # Opciones en BYMA liquidan T+1; acciones T+2
        plazo = "t1" if order.symbol.upper().startswith("GFG") else "t2"
        return {
            "mercado":      _MARKET,
            "simbolo":      order.symbol.upper(),
            "cantidad":     order.quantity,
            "precio":       order.limit_price,
            "plazo":        plazo,
            "validez":      "ParaElDia",
            "precioDeOrden": "Limite",
        }

    def _headers(self) -> dict[str, str]:
        h = self._token_mgr.get_headers()
        h["Content-Type"] = "application/json"
        return h

    def _post(self, path: str, body: dict) -> Optional[dict]:
        url = f"{_BASE_URL}/{path}"
        resp = self._session.post(url, json=body, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {}

    def _get(self, path: str) -> Optional[list]:
        url = f"{_BASE_URL}/{path}"
        try:
            resp = self._session.get(url, headers=self._headers(), timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("[IOL] GET %s falló: %s", path, exc)
            return None

    def _delete(self, path: str) -> None:
        url = f"{_BASE_URL}/{path}"
        resp = self._session.delete(url, headers=self._headers(), timeout=10)
        resp.raise_for_status()
