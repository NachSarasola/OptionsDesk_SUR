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
        dt = (datetime.now() - self.timestamp).total_seconds()
        return dt > max_age_s


@dataclass
class OptionsChain:
    """Snapshot completo de la cadena de opciones de GGAL."""

    underlying: str
    spot: Quote
    options: dict[str, Quote]      # symbol → Quote
    timestamp: datetime = field(default_factory=datetime.now)


class MarketDataProvider(ABC):
    """Interfaz para todas las fuentes de datos de mercado.

    Implementaciones:
      - HomeBrokerProvider: datos en tiempo real vía pyhomebroker (Bull Market)
      - BymaOpenProvider:   datos demorados de BYMA Open Data (fallback)
      - DemoProvider:       datos sintéticos para desarrollo y tests
    """

    @abstractmethod
    def get_options_chain(self) -> Optional[OptionsChain]:
        """Cadena de opciones GGAL más reciente. None si no hay datos."""
        ...

    @abstractmethod
    def get_spot(self) -> Optional[Quote]:
        """Quote de la acción GGAL."""
        ...

    @abstractmethod
    def get_caucion_tna(self, days: int = 30) -> Optional[float]:
        """TNA (%) de la caución colocadora para el plazo dado."""
        ...

    def is_connected(self) -> bool:
        return False
