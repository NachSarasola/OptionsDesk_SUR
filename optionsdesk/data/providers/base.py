"""Interfaz abstracta para todas las fuentes de datos de mercado."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Quote:
    """Quote de mercado para un instrumento."""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread_pct(self) -> float:
        m = self.mid
        if m <= 0:
            return 0.0
        return (self.ask - self.bid) / m * 100.0

    def is_stale(self, max_age_s: float = 300.0) -> bool:
        now = datetime.now(self.timestamp.tzinfo) if self.timestamp.tzinfo else datetime.now()
        dt = (now - self.timestamp).total_seconds()
        return dt > max_age_s


@dataclass
class OptionsChain:
    """Snapshot completo de la cadena de opciones de GGAL."""

    underlying: str
    spot: Quote
    options: dict[str, Quote]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketDataHealth:
    """Estado operativo del feed de mercado para diagnostico y UI."""

    source: str
    connected: bool
    last_success_ts: Optional[datetime] = None
    last_error: str = ""
    last_latency_ms: Optional[float] = None
    requests: int = 0
    timeouts: int = 0
    retries: int = 0
    options_seen: int = 0
    options_tradeable: int = 0


class MarketDataProvider(ABC):
    """Interfaz para todas las fuentes de datos de mercado."""

    @abstractmethod
    def get_options_chain(self) -> Optional[OptionsChain]:
        """Cadena de opciones GGAL mas reciente. None si no hay datos."""
        ...

    @abstractmethod
    def get_spot(self) -> Optional[Quote]:
        """Quote de la accion GGAL."""
        ...

    @abstractmethod
    def get_caucion_tna(self, days: int = 30) -> Optional[float]:
        """TNA (%) de la caucion colocadora para el plazo dado."""
        ...

    def is_connected(self) -> bool:
        return False

    def get_health(self) -> MarketDataHealth:
        return MarketDataHealth(source=self.__class__.__name__, connected=self.is_connected())
