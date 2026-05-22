"""Interfaz para la ejecución de órdenes.

v1 entrega solo PaperExecutor (registra órdenes, nunca envía nada).
Para escalar a automático: implementar un executor real (BullMarketExecutor
o IOLExecutor) que implemente esta misma interfaz, sin tocar el motor de señales.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    """Orden de compra/venta de un instrumento."""

    symbol: str
    side: OrderSide
    quantity: int               # número de lotes (cada lote = 100 acciones)
    limit_price: float          # precio límite en ARS
    strategy_ref: str           # ej. "COVERED_CALL:GFGC7000F"
    created_at: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    broker_id: Optional[str] = None
    notes: str = ""


class OrderExecutor(ABC):
    """Interfaz de ejecución de órdenes."""

    @abstractmethod
    def submit(self, order: Order) -> Order:
        """Envía la orden y devuelve el Order actualizado con su estado."""
        ...

    @abstractmethod
    def get_open_orders(self) -> list[Order]:
        ...

    @abstractmethod
    def cancel(self, order: Order) -> Order:
        ...
