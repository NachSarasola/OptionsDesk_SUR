"""Paper executor: registra órdenes propuestas sin enviarlas nunca.

Persiste un log JSONL en data/paper_orders.jsonl para revisar qué
habría ejecutado el bot. Útil para validar la lógica antes de operar real.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from optionsdesk.execution.base import Order, OrderExecutor, OrderStatus

logger = logging.getLogger(__name__)


class PaperExecutor(OrderExecutor):
    """Executor de paper trading — nunca envía órdenes reales."""

    def __init__(self, log_file: Optional[Path] = None) -> None:
        self._log_file = log_file or Path("data/paper_orders.jsonl")
        self._orders: list[Order] = []
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def submit(self, order: Order) -> Order:
        order.status = OrderStatus.SUBMITTED
        order.broker_id = f"PAPER-{len(self._orders):05d}"
        self._orders.append(order)
        self._append_log(order)
        logger.info(
            "[PAPER] %s %s x%d @ %.2f  ref=%s",
            order.side, order.symbol, order.quantity,
            order.limit_price, order.strategy_ref,
        )
        return order

    def get_open_orders(self) -> list[Order]:
        return [o for o in self._orders if o.status == OrderStatus.SUBMITTED]

    def cancel(self, order: Order) -> Order:
        order.status = OrderStatus.CANCELLED
        self._append_log(order)
        return order

    @property
    def order_count(self) -> int:
        return len(self._orders)

    def _append_log(self, order: Order) -> None:
        record = {
            "ts": order.created_at.isoformat(),
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.quantity,
            "price": order.limit_price,
            "status": order.status,
            "ref": order.strategy_ref,
        }
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
